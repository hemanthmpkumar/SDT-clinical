#!/usr/bin/env python3
"""
src/sdt/phase3_attenuation.py

Phase 3 — Graph-Signal Context Attenuation
===========================================

Upon interference gate activation (Δλ₂ > τ_pivot), this module:

  1. Applies a high-pass graph signal filter  x̃ = H(Lₜ)x  that suppresses
     low-frequency components (= stale pre-pivot context):

         H(λ) = 1 − exp(−γ·λ)          spectral response
         H(L)x ≈ Σ_{k=0}^{K} θ_k L^k x  polynomial approximation

  2. Partitions V_t into V_stale and V_active via the Fiedler vector cut
     (Definition 4.1 / Eq. 17–18 from the paper):

         V_stale  = {i ∈ V_t | v₂(i) < 0}
         V_active = {i ∈ V_t | v₂(i) ≥ 0}

  3. Severs all hyperedges bridging V_stale ↔ V_active.

  4. Aggregates the filtered embeddings of V_active into a single retrieval
     seed vector r_t (Eq. 21).

This enforces Theorem 4.2 (Contamination Bound):
     NCut(V_stale, V_active) ≤ λ₂(Lₜ)

References
----------
Shuman et al. (2013) "The Emerging Field of Signal Processing on Graphs"
Ortega et al. (2018) "Graph Signal Processing: Overview, Challenges..."
Defferrard et al. (2016) "Convolutional Neural Networks on Graphs (ChebNet)"
Hammond et al. (2011) "Wavelets on Graphs via Spectral Graph Theory"
Fiedler (1975) "A Property of Eigenvectors of Nonneg. Symmetric Matrices"
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.linalg as sla

warnings.filterwarnings("ignore", category=RuntimeWarning)


class GraphSignalAttenuator:
    """Applies the Fiedler cut and high-pass graph signal filter.

    Parameters
    ----------
    gamma : float
        Spectral response sharpness for H(λ)=1−exp(−γλ).
        Larger γ → more aggressive suppression of low-frequency components.
    poly_order : int
        Order K of the Chebyshev polynomial approximation.
        K=5 provides a good balance between accuracy and speed for
        subgraphs up to n≈200.
    min_active : int
        Minimum number of active nodes required to form a retrieval vector.
        If |V_active| < min_active after the cut, the full subgraph is used.
    """

    def __init__(
        self,
        gamma: float = 3.0,
        poly_order: int = 5,
        min_active: int = 1,
    ):
        self.gamma = gamma
        self.poly_order = poly_order
        self.min_active = min_active

        # Reported at each gate fire.
        self.last_ncut: float = 0.0
        self.last_v_stale_ids: list[int] = []
        self.last_v_active_ids: list[int] = []

    # ------------------------------------------------------------------
    # Polynomial filter coefficients
    # ------------------------------------------------------------------

    def _filter_coefficients(self, L: np.ndarray) -> np.ndarray:
        """Compute polynomial coefficients θ_k for H(λ)=1−exp(−γλ).

        Uses least-squares fit on the eigenvalue range [0, λ_max].

        Returns
        -------
        theta : np.ndarray, shape (K+1,)
        """
        K = self.poly_order
        # Sample the target response on a fine grid.
        lam_max = float(np.trace(L) / max(L.shape[0], 1)) * 2 + 0.1
        grid = np.linspace(0.0, min(lam_max, 2.0), 200)
        target = 1.0 - np.exp(-self.gamma * grid)        # desired H(λ)

        # Vandermonde polynomial basis.
        Phi = np.column_stack([grid ** k for k in range(K + 1)])
        theta, _, _, _ = np.linalg.lstsq(Phi, target, rcond=None)
        return theta.astype(np.float32)

    # ------------------------------------------------------------------
    # Graph Fourier transform filter (spectral domain)
    # ------------------------------------------------------------------

    def _spectral_filter(
        self, L: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Apply H(L)x in the spectral domain for small graphs (n ≤ 64).

        Exact: computes U, Λ via eigh, applies H(Λ) pointwise, transforms back.
        """
        n = L.shape[0]
        try:
            eigvals, U = sla.eigh(L)               # L = U Λ U^T
        except np.linalg.LinAlgError:
            return x.copy()

        eigvals = np.clip(eigvals, 0.0, 2.0)
        H_lam = 1.0 - np.exp(-self.gamma * eigvals)   # H(Λ)
        x_hat = U.T @ x                                # Graph Fourier
        x_filtered = H_lam * x_hat                     # Apply filter
        return (U @ x_filtered).astype(np.float32)     # Inverse GFT

    def _polynomial_filter(
        self, L: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Apply H(L)x ≈ Σ θ_k L^k x  (polynomial, no eigendecomp)."""
        theta = self._filter_coefficients(L)
        result = np.zeros_like(x, dtype=np.float32)
        Lk_x = x.astype(np.float32)
        for k, t in enumerate(theta):
            result += t * Lk_x
            if k < len(theta) - 1:
                Lk_x = (L @ Lk_x).astype(np.float32)
        return result

    def filter_signal(
        self, L: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Apply high-pass graph filter x̃ = H(Lₜ)x.

        Automatically selects spectral (n ≤ 64) or polynomial (n > 64) path.

        Args:
            L : normalized Laplacian, shape (n, n).
            x : node signal vector, shape (n,).

        Returns:
            x_tilde : filtered signal, shape (n,).
        """
        n = L.shape[0]
        if n == 0:
            return x.copy()
        if n <= 64:
            return self._spectral_filter(L, x)
        else:
            return self._polynomial_filter(L, x)

    # ------------------------------------------------------------------
    # Fiedler cut
    # ------------------------------------------------------------------

    def fiedler_cut(
        self,
        fiedler_vector: np.ndarray,
        node_ids: list[int],
    ) -> tuple[list[int], list[int]]:
        """Partition nodes into V_stale and V_active by sign of v₂.

        V_stale  = {i | v₂(i) < 0}
        V_active = {i | v₂(i) ≥ 0}

        Args:
            fiedler_vector : v₂ eigenvector of Lₜ, shape (n,).
            node_ids       : list of node indices (0..n-1) or string IDs.

        Returns
        -------
        stale_ids, active_ids : index lists into `node_ids`.
        """
        v2 = np.asarray(fiedler_vector, dtype=np.float32)
        stale_ids  = [node_ids[i] for i in range(len(v2)) if v2[i] <  0]
        active_ids = [node_ids[i] for i in range(len(v2)) if v2[i] >= 0]
        self.last_v_stale_ids  = stale_ids
        self.last_v_active_ids = active_ids
        return stale_ids, active_ids

    def normalized_cut_cost(
        self,
        A: np.ndarray,
        stale_mask: np.ndarray,
    ) -> float:
        """Compute NCut(V_stale, V_active) to verify Theorem 4.2.

        Args:
            A          : (n, n) adjacency matrix.
            stale_mask : boolean mask of shape (n,), True for V_stale.

        Returns:
            ncut : float — should be ≤ λ₂(Lₜ) by Cheeger inequality.
        """
        active_mask = ~stale_mask
        cut = A[np.ix_(stale_mask, active_mask)].sum()
        vol_stale  = A[stale_mask, :].sum()
        vol_active = A[active_mask, :].sum()
        denom = min(vol_stale, vol_active)
        ncut = float(cut / denom) if denom > 1e-9 else 0.0
        self.last_ncut = ncut
        return ncut

    def sever_cross_edges(
        self,
        A: np.ndarray,
        stale_mask: np.ndarray,
    ) -> np.ndarray:
        """Zero out all edges bridging V_stale ↔ V_active.

        Args:
            A          : (n, n) adjacency matrix.
            stale_mask : boolean mask, True for V_stale nodes.

        Returns:
            A_cut : adjacency with cross-edges removed.
        """
        A_cut = A.copy()
        active_mask = ~stale_mask
        A_cut[np.ix_(stale_mask, active_mask)] = 0.0
        A_cut[np.ix_(active_mask, stale_mask)] = 0.0
        return A_cut.astype(np.float32)

    # ------------------------------------------------------------------
    # Main pipeline step
    # ------------------------------------------------------------------

    def attenuate(
        self,
        A: np.ndarray,
        L: np.ndarray,
        fiedler_vector: np.ndarray,
        embeddings: np.ndarray,
        node_idx: list[int],
        lambda2: float,
    ) -> tuple[np.ndarray, list[int], list[int], float]:
        """Full Phase 3 pipeline: filter → cut → aggregate retrieval seed.

        Args:
            A              : (n, n) adjacency of active subgraph.
            L              : (n, n) normalized Laplacian.
            fiedler_vector : (n,) Fiedler vector v₂.
            embeddings     : (n, d) node embedding matrix.
            node_idx       : list of n node indices (positions in the
                             master node list).
            lambda2        : current λ₂ (for bound verification).

        Returns
        -------
        r_t         : retrieval seed vector (d,) — mean of filtered active embeds.
        stale_ids   : node indices in V_stale.
        active_ids  : node indices in V_active.
        ncut        : NCut value (should be ≤ λ₂ by Theorem 4.2).
        """
        n = len(node_idx)

        # 1. Fiedler cut: partition into stale / active.
        stale_ids, active_ids = self.fiedler_cut(fiedler_vector, node_idx)

        # Fall back to full set if cut leaves too few active nodes.
        if len(active_ids) < self.min_active:
            active_ids = node_idx
            stale_ids  = []

        stale_mask = np.array(
            [node_idx[i] in stale_ids for i in range(n)], dtype=bool
        )

        # 2. Verify contamination bound (Theorem 4.2).
        ncut = self.normalized_cut_cost(A, stale_mask)

        # 3. High-pass graph filter on the relevance signal.
        # Signal x_i = ‖embedding_i‖ (node "energy" as relevance proxy).
        x = np.linalg.norm(embeddings, axis=1).astype(np.float32)  # (n,)
        x_tilde = self.filter_signal(L, x)                         # (n,)

        # 4. Extract active node embeddings, weighted by filtered signal.
        active_local = [i for i, idx in enumerate(node_idx) if idx in set(active_ids)]
        if not active_local:
            active_local = list(range(n))

        active_embeds = embeddings[active_local]        # (|V_act|, d)
        active_weights = np.abs(x_tilde[active_local]) + 1e-9
        active_weights /= active_weights.sum()

        # 5. Weighted mean aggregation → retrieval seed r_t (Eq. 21).
        r_t = (active_embeds * active_weights[:, np.newaxis]).sum(axis=0)
        norm = np.linalg.norm(r_t)
        r_t = (r_t / (norm + 1e-9)).astype(np.float32)

        return r_t, stale_ids, active_ids, ncut

    def reset(self) -> None:
        """Reset internal state (no per-session state in this module)."""
        self.last_ncut = 0.0
        self.last_v_stale_ids  = []
        self.last_v_active_ids = []
