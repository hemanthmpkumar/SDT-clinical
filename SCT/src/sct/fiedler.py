"""
Phase 2 — Spectral Pivot Detection (SCT)
=========================================
Monitors algebraic connectivity λ₂(Δ_t) and declares paradigm pivots
when Δλ₂ > τ_pivot (adaptive or fixed threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .hypergraph import MultiModalHypergraph


@dataclass
class PivotEvent:
    """Record of a detected paradigm pivot."""
    step: int
    lambda2_before: float
    lambda2_after: float
    delta_lambda2: float
    fiedler_vector: np.ndarray
    node_ids: List[str]


class ScientificPivotDetector:
    """
    Monitors the normalized hypergraph Laplacian's Fiedler value λ₂(Δ_t).

    A paradigm pivot is declared when:
        Δλ₂(t) = λ₂(Δ_{t-1}) - λ₂(Δ_t) > τ_pivot(t)

    Supports both fixed and adaptive (rolling Gaussian) thresholds.

    Parameters
    ----------
    tau_pivot : float
        Fixed pivot threshold (used if adaptive=False).
    adaptive : bool
        Use rolling mean + z_alpha * std as adaptive threshold.
    z_alpha : float
        z-score for adaptive threshold (default 2.326 → α=0.01).
    warmup_steps : int
        Steps before adaptive threshold activates.
    """

    def __init__(
        self,
        tau_pivot: float = 0.15,
        adaptive: bool = True,
        z_alpha: float = 2.326,
        warmup_steps: int = 2,
    ):
        self.tau_pivot = tau_pivot
        self.adaptive = adaptive
        self.z_alpha = z_alpha
        self.warmup_steps = warmup_steps

        self._lambda2_history: List[float] = []
        self._delta_history: List[float] = []
        self._lambda2_prev: float = 0.0
        self._fiedler_vec_prev: Optional[np.ndarray] = None
        self._step: int = 0
        self.pivot_events: List[PivotEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, graph: MultiModalHypergraph) -> Tuple[bool, float, float]:
        """
        Compute λ₂ from the current hypergraph Laplacian and check for pivot.

        Returns
        -------
        is_pivot : bool
        lambda2 : float   (current)
        delta_lambda2 : float
        """
        Delta, node_ids = graph.build_laplacian()
        n = Delta.shape[0]

        lambda2, fiedler_vec = self._compute_fiedler(Delta, n)

        delta = self._lambda2_prev - lambda2
        self._lambda2_history.append(lambda2)
        self._delta_history.append(delta)

        threshold = self._get_threshold()
        is_pivot = (abs(delta) > threshold) and (self._step >= self.warmup_steps)

        if is_pivot:
            event = PivotEvent(
                step=self._step,
                lambda2_before=self._lambda2_prev,
                lambda2_after=lambda2,
                delta_lambda2=delta,
                fiedler_vector=fiedler_vec,
                node_ids=node_ids,
            )
            self.pivot_events.append(event)

        self._lambda2_prev = lambda2
        self._fiedler_vec_prev = fiedler_vec
        self._step += 1

        return is_pivot, lambda2, delta

    @property
    def lambda2_history(self) -> List[float]:
        return self._lambda2_history

    @property
    def delta_history(self) -> List[float]:
        return self._delta_history

    @property
    def last_fiedler_vector(self) -> Optional[np.ndarray]:
        return self._fiedler_vec_prev

    def contamination_bound(self, delta_lambda2: float) -> float:
        """
        Theorem 6.2: ε_cont ≤ 0.08 * (τ_pivot / Δλ₂)^{1/2}
        """
        tau = self._get_threshold()
        if delta_lambda2 <= 0:
            return 1.0
        return 0.08 * np.sqrt(tau / delta_lambda2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_fiedler(
        self, Delta: sp.csr_matrix, n: int
    ) -> Tuple[float, np.ndarray]:
        """Compute λ₂ and v₂ of Delta via ARPACK, warm-started."""
        if n <= 2:
            return 0.0, np.ones(n) / np.sqrt(max(n, 1))

        # Ensure numerical symmetry
        Delta_sym = (Delta + Delta.T) / 2.0

        # Dense computation for n <= 300 is exact, instantaneous (<1ms), and immune to ARPACK SM quirks
        if n <= 300:
            D = Delta_sym.toarray()
            vals, vecs = np.linalg.eigh(D)
            idx = np.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
            lambda2 = float(max(vals[1] if len(vals) > 1 else 0.0, 0.0))
            fiedler_vec = vecs[:, 1] if len(vals) > 1 else np.ones(n) / np.sqrt(n)
            return lambda2, fiedler_vec

        # ARPACK for large graphs (n > 300)
        try:
            k = min(3, n - 1)
            eigenvalues, eigenvectors = spla.eigsh(
                Delta_sym, k=k, which="SM",
                v0=v0, maxiter=500, tol=1e-6
            )
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            lambda2 = float(max(eigenvalues[min(1, len(eigenvalues)-1)], 0.0))
            fiedler_vec = eigenvectors[:, min(1, eigenvectors.shape[1]-1)]
        except Exception:
            lambda2 = self._lambda2_prev
            fiedler_vec = (self._fiedler_vec_prev if self._fiedler_vec_prev is not None
                           else np.ones(n) / np.sqrt(n))

        return lambda2, fiedler_vec

    def _get_threshold(self) -> float:
        valid_deltas = self._delta_history[self.warmup_steps:]
        if not self.adaptive or len(valid_deltas) < 3:
            return self.tau_pivot
        arr = np.array(valid_deltas[-20:])
        mu = float(np.mean(arr))
        sigma = float(np.std(arr)) + 1e-8
        adaptive_thresh = mu + self.z_alpha * sigma
        return float(np.clip(adaptive_thresh, 0.03, self.tau_pivot))
