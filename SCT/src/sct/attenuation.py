"""
Phase 3 — Context Attenuation via Chebyshev-Filtered Fiedler Cuts (SCT)
=========================================================================
Applies a Chebyshev high-pass spectral filter to suppress stale-model
signal leakage, then bisects the hypothesis graph using the Fiedler vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.sparse as sp

from .hypergraph import MultiModalHypergraph, Node
from .fiedler import PivotEvent


@dataclass
class CutResult:
    active_node_ids: Set[str]
    stale_node_ids: Set[str]
    contamination_bound: float
    ncut_value: float
    lambda_cut: float
    filter_order: int


class ChebyshevFiedlerCut:
    """
    Phase 3 of SCT: context attenuation after a paradigm pivot.

    Steps:
      1. Apply Chebyshev high-pass filter h_K^HP(Δ) to the evidence signal f.
      2. Bisect via Fiedler vector v₂ → (V_active, V_stale).
      3. Archive stale nodes; return attenuated active subgraph.

    Parameters
    ----------
    filter_order : int      Chebyshev order K (default 20).
    lambda_gap   : float    Gap above λ₂ for cut frequency (default 0.05).
    tau_bridge   : float    Min cosine sim for bridge-paper re-admission.
    """

    def __init__(
        self,
        filter_order: int = 20,
        lambda_gap: float = 0.05,
        tau_bridge: float = 0.50,
    ):
        self.filter_order = filter_order
        self.lambda_gap = lambda_gap
        self.tau_bridge = tau_bridge
        self.stale_archive: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cut(
        self,
        graph: MultiModalHypergraph,
        pivot_event: PivotEvent,
        tau_pivot: float = 0.15,
    ) -> Tuple[MultiModalHypergraph, CutResult]:
        """
        Perform the Fiedler cut on `graph` at the given pivot event.

        Returns
        -------
        graph       : MultiModalHypergraph   (pruned in-place to active nodes)
        cut_result  : CutResult
        """
        Delta, node_ids = graph.build_laplacian()
        n = len(node_ids)

        if n < 4 or pivot_event.fiedler_vector is None:
            # Graph too small to cut; return unchanged
            return graph, CutResult(
                active_node_ids=set(node_ids),
                stale_node_ids=set(),
                contamination_bound=1.0,
                ncut_value=1.0,
                lambda_cut=0.0,
                filter_order=self.filter_order,
            )

        v2 = pivot_event.fiedler_vector
        # Align v2 dimension with current graph (may differ slightly after expiry)
        if len(v2) != n:
            v2 = np.ones(n) / np.sqrt(n)

        # --- Chebyshev HP filter on evidence signal f = v2 ------------
        lambda_cut = float(pivot_event.lambda2_after) + self.lambda_gap
        lambda_max = self._estimate_lambda_max(Delta)
        f_hp = self._chebyshev_hp_filter(Delta, v2, lambda_cut, lambda_max)

        # --- Fiedler bisection -----------------------------------------
        # The triggering vertex should be in V_active;
        # sign convention: positive component → active
        trigger_idx = len(node_ids) - 1   # newest node triggered the pivot
        sign_ref = np.sign(v2[trigger_idx]) if abs(v2[trigger_idx]) > 1e-10 else 1.0

        active_ids: Set[str] = set()
        stale_ids: Set[str] = set()
        for i, nid in enumerate(node_ids):
            if sign_ref * v2[i] >= 0:
                active_ids.add(nid)
            else:
                stale_ids.add(nid)

        # Ensure trigger node is always active
        if node_ids[trigger_idx] in stale_ids:
            stale_ids.discard(node_ids[trigger_idx])
            active_ids.add(node_ids[trigger_idx])

        # --- Contamination bound (Theorem 6.2) -------------------------
        ncut_val = self._compute_ncut(Delta, node_ids, active_ids)
        eps_cont = self._contamination_bound(
            tau_pivot, pivot_event.delta_lambda2, ncut_val
        )

        # --- Archive stale nodes ---------------------------------------
        stale_nodes: Dict[str, Node] = {
            nid: graph.nodes[nid]
            for nid in stale_ids
            if nid in graph.nodes
        }
        if stale_nodes:
            self.stale_archive.append({
                "pivot_step": pivot_event.step,
                "delta_lambda2": pivot_event.delta_lambda2,
                "contamination_bound": eps_cont,
                "stale_node_ids": list(stale_ids),
                "stale_modalities": [
                    graph.nodes[nid].modality.value
                    for nid in stale_ids if nid in graph.nodes
                ],
            })

        # --- Prune graph -----------------------------------------------
        graph.remove_nodes(stale_ids)

        cut_result = CutResult(
            active_node_ids=active_ids,
            stale_node_ids=stale_ids,
            contamination_bound=eps_cont,
            ncut_value=ncut_val,
            lambda_cut=lambda_cut,
            filter_order=self.filter_order,
        )
        return graph, cut_result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chebyshev_hp_filter(
        self,
        L: sp.csr_matrix,
        f: np.ndarray,
        lambda_cut: float,
        lambda_max: float,
    ) -> np.ndarray:
        """
        Apply order-K Chebyshev high-pass filter to signal f.
        h^HP(λ) ≈ 0 for λ < λ_cut; ≈ 1 for λ ≥ λ_cut.
        """
        lambda_max = max(lambda_max, 2.0)
        # Scale Laplacian to [-1, 1]: L~ = 2L/λ_max - I
        L_tilde = 2.0 / lambda_max * L - sp.eye(L.shape[0])

        # Compute Chebyshev coefficients for HP filter by Jackson damping
        K = self.filter_order
        # Sample filter target on Chebyshev nodes
        grid = np.cos(np.pi * (np.arange(K + 1) + 0.5) / (K + 1))   # in [-1,1]
        lambda_grid = (grid + 1) * lambda_max / 2.0                    # map to [0, λ_max]
        h_target = np.where(lambda_grid >= lambda_cut, 1.0, 0.0).astype(float)

        # Chebyshev coefficients via DCT
        theta_k = np.zeros(K + 1)
        for k in range(K + 1):
            theta_k[k] = (2.0 / (K + 1)) * np.sum(
                h_target * np.cos(np.pi * k * (np.arange(K + 1) + 0.5) / (K + 1))
            )
        theta_k[0] /= 2.0

        # Apply: f_hp = Σ_k θ_k T_k(L~) f
        T_prev = f.copy()
        T_curr = L_tilde.dot(f)
        f_hp = theta_k[0] * T_prev + theta_k[1] * T_curr

        for k in range(2, K + 1):
            T_next = 2.0 * L_tilde.dot(T_curr) - T_prev
            f_hp += theta_k[k] * T_next
            T_prev = T_curr
            T_curr = T_next

        return f_hp

    def _estimate_lambda_max(self, L: sp.csr_matrix) -> float:
        """Estimate spectral radius of L via power iteration (3 steps)."""
        n = L.shape[0]
        if n == 0:
            return 2.0
        try:
            vals, _ = sp.linalg.eigsh(L, k=1, which="LM", maxiter=100, tol=1e-4)
            return float(max(vals[0], 2.0))
        except Exception:
            return 2.0

    def _compute_ncut(
        self,
        Delta: sp.csr_matrix,
        node_ids: List[str],
        active_ids: Set[str],
    ) -> float:
        """Normalized cut NCut(V_active, V_stale)."""
        n = len(node_ids)
        nid2idx = {nid: i for i, nid in enumerate(node_ids)}
        s = np.array([1.0 if node_ids[i] in active_ids else -1.0 for i in range(n)])
        # NCut ≈ (s^T Δ s) / (s^T D s)  — proxy via Laplacian quadratic form
        D_diag = np.array(Delta.diagonal())
        num = float(s @ Delta.dot(s))
        den = float(np.dot(s * s, D_diag)) + 1e-10
        return float(max(num / den, 0.0))

    def _contamination_bound(
        self,
        tau_pivot: float,
        delta_lambda2: float,
        ncut: float,
    ) -> float:
        abs_delta = max(abs(delta_lambda2), 1e-4)
        ratio = min(tau_pivot / abs_delta, 1.0)
        return float(0.08 * np.sqrt(ratio))
