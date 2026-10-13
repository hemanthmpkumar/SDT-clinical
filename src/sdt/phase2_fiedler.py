#!/usr/bin/env python3
"""
src/sdt/phase2_fiedler.py

Phase 2 — Spectral Pivot Detection
====================================

Monitors the second smallest eigenvalue λ₂(Lₜ) of the normalized graph
Laplacian of the active subgraph Gₜ.  The Fiedler value is a continuous
measure of the diagnostic hypothesis' structural cohesion:

    • Growing λ₂   →  evidence accumulation, increasing connectivity.
    • Sudden drop  →  the graph is fracturing — a diagnostic pivot.

Interference gate trigger (Eq. 14 from paper):

    Δλ₂(t) = λ₂(L_{t-1}) − λ₂(Lₜ) > τ_pivot

This replaces GDT's geodesic shift ratio κₜ with a topology-native,
computationally efficient O(|Eₜ|) trigger robust to multi-modal noise.

References
----------
Fiedler (1973) "Algebraic Connectivity of Graphs"
Chung (1997)   "Spectral Graph Theory"
Spielman & Teng (2007) "Spectral Partitioning Works"
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Optional

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Lazy torch import for GPU acceleration
_torch_phase2 = None
_torch_available_phase2 = False

def _import_torch_phase2():
    """Lazy import torch for Phase 2 GPU acceleration."""
    global _torch_phase2, _torch_available_phase2
    if not _torch_available_phase2:
        try:
            import torch
            _torch_phase2 = torch
            _torch_available_phase2 = True
        except ImportError:
            _torch_available_phase2 = False
    return _torch_available_phase2

def _get_device_phase2():
    """Get best available device for Phase 2."""
    if not _import_torch_phase2():
        return None
    if hasattr(_torch_phase2.backends, 'mps') and _torch_phase2.backends.mps.is_available():
        return _torch_phase2.device('mps')
    if _torch_phase2.cuda.is_available():
        return _torch_phase2.device('cuda')
    return None

warnings.filterwarnings("ignore", category=RuntimeWarning)


class SpectralPivotDetector:
    """Real-time tracker of λ₂(Lₜ) with interference gate.

    Parameters
    ----------
    tau_pivot : float
        Threshold Δλ₂ above which a diagnostic pivot is declared.
        Smaller values → more sensitive; larger values → fewer false alarms.
        Empirically tuned to τ=0.15 on MIMIC-IV query logs.
    history_len : int
        Number of past λ₂ values retained for trend analysis.
    min_nodes : int
        Minimum active nodes required before monitoring starts
        (prevents spurious triggers on trivially small subgraphs).
    """

    def __init__(
        self,
        tau_pivot: float = 0.15,
        history_len: int = 20,
        min_nodes: int = 3,
    ):
        self.tau_pivot = tau_pivot
        self.history_len = history_len
        self.min_nodes = min_nodes

        self._lambda2_history: deque[float] = deque(maxlen=history_len)
        self._laplacian_history: list[Optional[np.ndarray]] = []

        # Per-session state
        self.last_lambda2: float = 0.0
        self.last_fiedler_vector: Optional[np.ndarray] = None
        self.last_delta_lambda2: float = 0.0
        self.gate_triggered: bool = False
        self.n_triggers: int = 0
        self.trigger_times: list[int] = []          # turn indices of triggers
        self._turn: int = 0

    # ------------------------------------------------------------------
    # Laplacian computation
    # ------------------------------------------------------------------

    @staticmethod
    def normalized_laplacian(A: np.ndarray) -> np.ndarray:
        """Compute the normalized Laplacian Lₜ = I − D^{-½} A D^{-½}.

        Uses GPU matmul when available for large matrices (>100 nodes).

        Args:
            A : symmetric adjacency matrix of shape (n, n).

        Returns:
            L : normalized Laplacian, shape (n, n).
        """
        n = A.shape[0]
        d = A.sum(axis=1)                              # degree vector
        d_inv_sqrt = np.where(d > 1e-9, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        # GPU acceleration for large matrices
        if _torch_available_phase2 and n > 100:
            device = _get_device_phase2()
            if device is not None:
                try:
                    torch = _torch_phase2
                    A_t = torch.tensor(A, device=device)
                    D_t = torch.tensor(D_inv_sqrt, device=device)
                    prod = torch.matmul(D_t, torch.matmul(A_t, D_t))
                    L = (torch.eye(n, device=device) - prod).cpu().numpy()
                    return L.astype(np.float32)
                except Exception:
                    pass  # Fall back to numpy
        
        L = np.eye(n, dtype=np.float32) - (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)
        return L

    @staticmethod
    def fiedler_value_and_vector(
        L: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute λ₂(L) and its eigenvector v₂ (the Fiedler vector).

        Uses GPU eigenvalue decomposition when available for large matrices.
        Falls back to scipy.linalg.eigh for small dense subgraphs (n ≤ 512).
        For larger graphs, falls back to ARPACK sparse eigsh.

        Returns
        -------
        lambda2 : float
            Second smallest eigenvalue.
        v2 : np.ndarray, shape (n,)
            Corresponding Fiedler eigenvector.
        """
        n = L.shape[0]
        if n <= 1:
            return 0.0, np.ones(n, dtype=np.float32)

        try:
            if n <= 512:
                # Dense path: try GPU first for large enough matrices, then scipy
                if _torch_available_phase2 and n > 200:
                    device = _get_device_phase2()
                    if device is not None:
                        try:
                            torch = _torch_phase2
                            L_t = torch.tensor(L, device=device)
                            # eigh returns (values, vectors) for symmetric matrix
                            eigvals, eigvecs = torch.linalg.eigh(L_t)
                            # Take first 2 eigenvalues/vectors
                            eigvals = eigvals[:2]
                            eigvecs = eigvecs[:, :2]
                            # Sort ascending
                            order = torch.argsort(eigvals)
                            eigvals = eigvals[order]
                            eigvecs = eigvecs[:, order]
                            lambda2 = float(torch.clamp(eigvals[1], 0.0, 2.0).cpu().item())
                            v2 = eigvecs[:, 1].cpu().numpy().astype(np.float32)
                            return lambda2, v2
                        except Exception:
                            pass  # Fall back to scipy below
                
                # Dense path: full eigendecomposition, take k=2.
                eigvals, eigvecs = sla.eigh(L, subset_by_index=[0, 1])
            else:
                # Sparse path for large subgraphs.
                L_sp = sp.csr_matrix(L)
                eigvals, eigvecs = spla.eigsh(L_sp, k=2, which="SM", tol=1e-6)
                # Sort ascending.
                order = np.argsort(eigvals)
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]

            lambda2 = float(np.clip(eigvals[1], 0.0, 2.0))
            v2 = eigvecs[:, 1].astype(np.float32)
        except (np.linalg.LinAlgError, spla.ArpackNoConvergence):
            lambda2 = 0.0
            v2 = np.ones(n, dtype=np.float32) / np.sqrt(n)

        return lambda2, v2

    # ------------------------------------------------------------------
    # Main update step
    # ------------------------------------------------------------------

    def update(
        self, A: np.ndarray
    ) -> tuple[bool, float, Optional[np.ndarray]]:
        """Process a new adjacency matrix and check the interference gate.

        Call this once per turn after building the hypergraph adjacency.

        Args:
            A : float array of shape (n, n), symmetric adjacency matrix
                of the active subgraph Gₜ.

        Returns
        -------
        triggered : bool
            True if the Fiedler gate fired (Δλ₂ > τ_pivot).
        lambda2 : float
            Current λ₂(Lₜ).
        fiedler_vector : np.ndarray or None
            Fiedler vector v₂ (useful for Phase 3 cut). None if n < 2.
        """
        self._turn += 1
        n = A.shape[0]

        if n < self.min_nodes:
            self.gate_triggered = False
            return False, self.last_lambda2, self.last_fiedler_vector

        # Compute Laplacian and Fiedler value.
        L = self.normalized_laplacian(A)
        lambda2, v2 = self.fiedler_value_and_vector(L)

        # Compute Δλ₂ and evaluate gate.
        delta = self.last_lambda2 - lambda2
        self.last_delta_lambda2 = delta

        triggered = (
            len(self._lambda2_history) >= 2         # need history to compare
            and delta > self.tau_pivot
        )

        if triggered:
            self.n_triggers += 1
            self.trigger_times.append(self._turn)

        self.gate_triggered = triggered
        self.last_lambda2 = lambda2
        self.last_fiedler_vector = v2
        self._lambda2_history.append(lambda2)

        return triggered, lambda2, v2

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def lambda2_trend(self) -> Optional[np.ndarray]:
        """Return the recent λ₂ history as a numpy array, or None."""
        if len(self._lambda2_history) < 2:
            return None
        return np.array(self._lambda2_history, dtype=np.float32)

    def algebraic_connectivity_report(self) -> dict:
        """Return a structured snapshot of the current spectral state."""
        return {
            "turn": self._turn,
            "lambda2": self.last_lambda2,
            "delta_lambda2": self.last_delta_lambda2,
            "tau_pivot": self.tau_pivot,
            "gate_triggered": self.gate_triggered,
            "n_triggers": self.n_triggers,
            "lambda2_history": list(self._lambda2_history),
        }

    def reset(self) -> None:
        """Clear all per-session state."""
        self._lambda2_history.clear()
        self._laplacian_history.clear()
        self.last_lambda2 = 0.0
        self.last_fiedler_vector = None
        self.last_delta_lambda2 = 0.0
        self.gate_triggered = False
        self.n_triggers = 0
        self.trigger_times.clear()
        self._turn = 0
