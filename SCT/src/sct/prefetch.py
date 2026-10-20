"""
Phase 4 — Adaptive Observation Prefetching (SCT)
=================================================
Spectrally biased random walk over the global observational graph,
constrained by a scientific cognitive load budget (knapsack).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .hypergraph import MultiModalHypergraph, Node, ModalityType


# Cognitive load cost per observation type (c(ω) in proposal)
OBSERVATION_COST = {
    ModalityType.SPECTRAL:    1.0,   # JWST transit visit
    ModalityType.PHOTOMETRIC: 0.5,   # Ground/space photometry
    ModalityType.IMAGING:     1.5,   # Coronagraphy / MIRI imaging
    ModalityType.MODEL:       0.3,   # Neural emulator retrieval
    ModalityType.LITERATURE:  0.05,  # Literature synthesis
}


@dataclass
class PrefetchCandidate:
    node: Node
    info_gain: float
    cost: float
    ratio: float          # info_gain / cost
    rank: int = 0


class AdaptiveObservationPrefetcher:
    """
    Phase 4 of SCT: select the next most informative observations.

    After a paradigm pivot and Fiedler cut, we:
      1. Compute spectral information gain InfoGain_λ(v_j) for all
         candidate nodes in the global graph.
      2. Run a spectrally biased random walk to generate a ranked list.
      3. Apply greedy knapsack to select within budget B_sci.

    Parameters
    ----------
    budget : float        Scientific cognitive load budget B_sci.
    beta_spec : float     Spectral bias temperature β_spec.
    walk_length : int     Number of random walk steps T_walk.
    """

    def __init__(
        self,
        budget: float = 10.0,
        beta_spec: float = 2.0,
        walk_length: int = 100,
    ):
        self.budget = budget
        self.beta_spec = beta_spec
        self.walk_length = walk_length

    def prefetch(
        self,
        active_graph: MultiModalHypergraph,
        candidate_nodes: List[Node],
        lambda2: float,
        fiedler_vec: Optional[np.ndarray],
    ) -> List[PrefetchCandidate]:
        """
        Rank candidate (unobserved) nodes by spectral information gain
        and return a budget-feasible prefetch list.

        Parameters
        ----------
        active_graph     : current attenuated active subgraph
        candidate_nodes  : pool of potential next observations
        lambda2          : current λ₂ (post-pivot)
        fiedler_vec      : current Fiedler vector v₂ (n_active,)

        Returns
        -------
        prefetch_list : List[PrefetchCandidate]   (ranked, budget-feasible)
        """
        if not candidate_nodes:
            return []

        active_embs, active_ids = active_graph.get_node_embeddings_matrix()
        n_active = len(active_ids)

        # --- Compute InfoGain_λ for each candidate --------------------
        candidates: List[PrefetchCandidate] = []
        for node in candidate_nodes:
            ig = self._info_gain(
                node, active_embs, fiedler_vec, lambda2, n_active
            )
            cost = OBSERVATION_COST.get(node.modality, 1.0)
            candidates.append(PrefetchCandidate(
                node=node,
                info_gain=ig,
                cost=cost,
                ratio=ig / max(cost, 1e-6),
            ))

        # --- Greedy knapsack ------------------------------------------
        candidates.sort(key=lambda c: c.ratio, reverse=True)
        selected: List[PrefetchCandidate] = []
        budget_remaining = self.budget

        for rank, cand in enumerate(candidates):
            if cand.cost <= budget_remaining:
                cand.rank = len(selected) + 1
                selected.append(cand)
                budget_remaining -= cand.cost
            if budget_remaining <= 0:
                break

        return selected

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _info_gain(
        self,
        node: Node,
        active_embs: np.ndarray,
        fiedler_vec: Optional[np.ndarray],
        lambda2: float,
        n_active: int,
    ) -> float:
        """
        Rank-1 Fiedler perturbation approximation:
          InfoGain_λ(v_j) ≈ (1/λ₂) Σ_i w²_ij [v₂]²_i

        where w_ij = predicted edge weight (cosine similarity proxy).
        """
        if n_active == 0 or active_embs.shape[0] == 0:
            return 0.0

        # Predicted edge weights: cosine similarity to each active node
        e_j = node.embedding
        norms_active = np.linalg.norm(active_embs, axis=1) + 1e-10
        norm_j = np.linalg.norm(e_j) + 1e-10
        cos_sims = (active_embs @ e_j) / (norms_active * norm_j)
        cos_sims = np.clip(cos_sims, 0, 1)

        if fiedler_vec is not None and len(fiedler_vec) == n_active:
            v2_sq = fiedler_vec ** 2
        else:
            v2_sq = np.ones(n_active) / n_active

        lam2 = max(lambda2, 1e-4)
        info_gain = float(np.sum(cos_sims ** 2 * v2_sq) / lam2)

        # Spectral bias modulation
        info_gain_biased = np.exp(self.beta_spec * info_gain) * info_gain
        return float(info_gain_biased)
