#!/usr/bin/env python3
"""
src/sdt/phase4_prefetch.py

Phase 4 — Anticipatory Prefetch Engine & Simulation Interface
=============================================================

Implements the optimal-control prefetch engine (Eq. 23–26 from paper):

    J(u_t) = E[α·L_delay(G_{t+1}, u_t) − β·R_struct(u_t) + γ·Ω(u_t)]

The engine has three components:

  1. Spectral Node Embedding  φ(v) ∈ R^m
     Projects each corpus node onto the leading m Laplacian eigenvectors
     (computed offline on G_global).  Captures long-range structural
     similarity beyond the local neighborhood.

  2. Spectrally-Biased Random Walk  (Eq. 22 from paper)
     Transition probability biased by spectral proximity:
         P(v_j|v_i) ∝ P₀(v_j|v_i) · exp(−τ‖φ(v_i)−φ(v_j)‖²)

  3. Cognitive-Load-Bounded Delivery  (Sweller constraint, Eq. 24)
     Greedy knapsack: select nodes by walk score while
         Σ H(v|Gₜ) ≤ C_max
     where H(v|Gₜ) = −log P(v|G_t^active) is the incremental cognitive burden.

References
----------
Page et al. (1999)     "The PageRank Citation Ranking"
Lovász (1993)          "Random Walks on Graphs: A Survey"
Grover & Leskovec (2016) "node2vec"
Sweller (1988)          "Cognitive Load During Problem Solving"
Sweller et al. (2019)   "Cognitive Architecture and Instructional Design: 20 Years Later"
Bertsekas (1995)        "Dynamic Programming and Optimal Control"
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.linalg as sla

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Global graph spectral embedding (computed once, offline)
# ---------------------------------------------------------------------------

class GlobalSpectralEmbedding:
    """Precomputes and caches spectral embeddings φ(v) for G_global.

    Parameters
    ----------
    m : int
        Embedding dimension (number of leading Laplacian eigenvectors).
    """

    def __init__(self, m: int = 16):
        self.m = m
        self._phi: Optional[np.ndarray] = None    # (N_global, m)
        self._node_ids: Optional[list] = None
        self._id_to_row: dict = {}
        self._fitted: bool = False

    def fit(
        self,
        A_global: np.ndarray,
        node_ids: list,
    ) -> "GlobalSpectralEmbedding":
        """Compute φ(v) = [u₂(v), …, u_{m+1}(v)] for all nodes.

        Args:
            A_global : (N, N) global adjacency matrix (may be dense).
            node_ids : ordered list of node identifiers.

        Returns:
            self
        """
        n = A_global.shape[0]
        self._node_ids = list(node_ids)
        self._id_to_row = {nid: i for i, nid in enumerate(node_ids)}

        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        if not sp.issparse(A_global):
            A_global = sp.csr_matrix(A_global)

        # Compute transition matrix M = D^{-1/2} A D^{-1/2}
        d = np.array(A_global.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(d > 1e-9, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        
        M_global = D_inv_sqrt @ A_global @ D_inv_sqrt

        k = min(self.m + 1, n - 2)
        try:
            # Finding the LARGEST eigenvalues of M is mathematically equivalent
            # to finding the SMALLEST eigenvalues of L = I - M, but ARPACK
            # converges exponentially faster on LA (Largest Algebraic) than SM.
            eigvals_M, eigvecs = spla.eigsh(M_global, k=k, which="LA", tol=1e-3)
            # The eigenvalues of L are 1 - eigvals(M). 
            # They are returned in ascending order of algebraic value of M, 
            # so we must reverse them to get ascending order of L!
            eigvals = 1.0 - eigvals_M[::-1]
            eigvecs = eigvecs[:, ::-1]
        except Exception as e:
            print(f"Warning: GlobalSpectralEmbedding eigsh failed ({e}). Falling back to zeros.")
            eigvals = np.zeros(k + 1)
            eigvecs = np.eye(n, k + 1)

        # φ(v) = eigenvectors 1..m (skip index 0, the constant vector).
        end_idx = min(self.m + 1, eigvecs.shape[1])
        phi = eigvecs[:, 1:end_idx].astype(np.float32)
        # Pad if needed.
        if phi.shape[1] < self.m:
            pad = np.zeros((n, self.m - phi.shape[1]), dtype=np.float32)
            phi = np.hstack([phi, pad])

        self._phi = phi          # (N, m)
        self._fitted = True
        return self

    def get(self, node_id) -> Optional[np.ndarray]:
        """Return the spectral embedding of a single node."""
        if not self._fitted:
            return None
        row = self._id_to_row.get(node_id)
        if row is None:
            return None
        return self._phi[row]

    def get_batch(self, node_ids: list) -> np.ndarray:
        """Return embeddings for a batch of node IDs, shape (len, m)."""
        out = []
        for nid in node_ids:
            phi = self.get(nid)
            out.append(phi if phi is not None else np.zeros(self.m, dtype=np.float32))
        return np.stack(out, axis=0)

    @property
    def fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Spectrally-biased random walk
# ---------------------------------------------------------------------------

class SpectralRandomWalk:
    """Runs spectrally-biased random walks on G_global.

    Parameters
    ----------
    tau : float
        Spectral bias strength. τ → 0 reduces to unbiased PageRank;
        τ → ∞ makes transitions purely spectral-similarity-driven.
    n_walk : int
        Number of walk steps per starting node.
    n_restarts : int
        Number of independent restarts per starting node (for variance reduction).
    """

    def __init__(self, tau: float = 2.0, n_walk: int = 30, n_restarts: int = 5):
        self.tau = tau
        self.n_walk = n_walk
        self.n_restarts = n_restarts

    def _transition_probs(
        self,
        v_i_phi: np.ndarray,
        neighbors_phi: np.ndarray,
        base_probs: np.ndarray,
    ) -> np.ndarray:
        """Compute biased P(v_j|v_i) for the neighbors of v_i (Eq. 22).

        Args:
            v_i_phi       : spectral embedding of current node, shape (m,).
            neighbors_phi : spectral embeddings of neighbors, shape (K, m).
            base_probs    : P₀(v_j|v_i) = A_ij/D_ii, shape (K,).

        Returns:
            probs : normalized transition probabilities, shape (K,).
        """
        diffs = neighbors_phi - v_i_phi[np.newaxis, :]      # (K, m)
        sq_dist = (diffs ** 2).sum(axis=1)                  # (K,)
        bias = np.exp(-self.tau * sq_dist)                  # (K,)
        raw = base_probs * bias                             # (K,)
        total = raw.sum()
        if total < 1e-12:
            return np.ones(len(raw)) / len(raw)
        return raw / total

    def walk(
        self,
        start_nodes: list,
        A: np.ndarray,
        phi: GlobalSpectralEmbedding,
        node_ids: list,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate random walks and accumulate visitation counts π(v).

        Args:
            start_nodes : IDs of starting nodes (from V_active).
            A           : (N, N) global adjacency matrix.
            phi         : fitted GlobalSpectralEmbedding.
            node_ids    : ordered list of all node IDs in G_global.
            rng         : numpy random generator.

        Returns:
            pi : visitation count vector, shape (N,), float32.
        """
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)
        pi = np.zeros(N, dtype=np.float32)

        for start_id in start_nodes:
            start_idx = id_to_idx.get(start_id)
            if start_idx is None:
                continue
            for _ in range(self.n_restarts):
                cur = start_idx
                for _ in range(self.n_walk):
                    row = A[cur]
                    neighbors = row.indices
                    if len(neighbors) == 0:
                        break
                    base_p = row.data / (row.data.sum() + 1e-12)
                    cur_phi = phi._phi[cur] if phi.fitted else np.zeros(phi.m)
                    nbr_phi = (
                        phi._phi[neighbors]
                        if phi.fitted
                        else np.zeros((len(neighbors), phi.m), dtype=np.float32)
                    )
                    probs = self._transition_probs(cur_phi, nbr_phi, base_p)
                    cur = int(rng.choice(neighbors, p=probs))
                    pi[cur] += 1.0

        return pi


# ---------------------------------------------------------------------------
# Cognitive Load model  (Sweller constraint)
# ---------------------------------------------------------------------------

class CognitiveLoadModel:
    """Models incremental cognitive burden H(v|Gₜ) = −log P(v|G_t^active).

    A simple but theoretically grounded estimate: the surprise of seeing
    node v given the current active context.  Nodes similar to the active
    context are cheap; novel nodes are costly.
    """

    def __init__(self, c_max: float = 7.0):
        """
        Args:
            c_max : Maximum total cognitive load budget (Miller's 7±2 rule).
        """
        self.c_max = c_max

    def node_cost(
        self,
        node_embed: np.ndarray,
        active_embeds: np.ndarray,
    ) -> float:
        """Estimate H(v|Gₜ).

        Uses negative log of mean cosine similarity to the active context.
        Floor at 0, ceiling at c_max per node.
        """
        if len(active_embeds) == 0:
            return self.c_max
        sims = active_embeds @ node_embed / (
            np.linalg.norm(active_embeds, axis=1) * np.linalg.norm(node_embed) + 1e-9
        )
        p = float(np.clip(sims.mean(), 1e-6, 1.0))
        cost = -np.log(p)
        return float(np.clip(cost, 0.0, self.c_max))

    def greedy_knapsack(
        self,
        candidates: list,           # list of (node_id, score, embed)
        active_embeds: np.ndarray,
    ) -> list:
        """Select the highest-scoring candidates while ΣH(v|Gₜ) ≤ C_max.

        Args:
            candidates    : list of (node_id, walk_score, embedding).
            active_embeds : (|V_active|, d) embedding matrix of current context.

        Returns:
            selected : list of (node_id, score) in descending score order.
        """
        # Sort by walk score descending.
        ranked = sorted(candidates, key=lambda x: -x[1])
        selected = []
        budget = self.c_max
        for node_id, score, embed in ranked:
            cost = self.node_cost(embed, active_embeds)
            if cost <= budget:
                selected.append((node_id, score))
                budget -= cost
            if budget <= 0:
                break
        return selected


# ---------------------------------------------------------------------------
# AnticipatoryPrefetchEngine — Phase 4 orchestrator
# ---------------------------------------------------------------------------

class AnticipatoryPrefetchEngine:
    """Orchestrates spectral walk + cognitive-load filtering for prefetching.

    Parameters
    ----------
    m : int
        Spectral embedding dimension for G_global.
    tau : float
        Spectral bias strength in random walk.
    n_walk : int
        Walk length per starting node.
    n_restarts : int
        Walk restarts per starting node.
    c_max : float
        Cognitive load budget (Sweller's constraint).
    top_k_candidates : int
        Number of top walk-scored candidates before cognitive filtering.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        m: int = 16,
        tau: float = 2.0,
        n_walk: int = 30,
        n_restarts: int = 5,
        c_max: float = 7.0,
        top_k_candidates: int = 50,
        seed: int = 42,
    ):
        self.m = m
        self.c_max = c_max
        self.top_k_candidates = top_k_candidates
        self._rng = np.random.default_rng(seed)

        self._global_embed = GlobalSpectralEmbedding(m=m)
        self._walker = SpectralRandomWalk(tau=tau, n_walk=n_walk, n_restarts=n_restarts)
        self._cog_model = CognitiveLoadModel(c_max=c_max)

        # Global graph cache (built once per corpus).
        self._A_global: Optional[np.ndarray] = None
        self._global_node_ids: Optional[list] = None
        self._global_node_embeds: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Global graph initialisation (offline, once per corpus)
    # ------------------------------------------------------------------

    def build_global_graph(
        self,
        node_ids: list,
        node_embeddings: np.ndarray,
        k_neighbors: int = 10,
    ) -> "AnticipatoryPrefetchEngine":
        """Build G_global as a k-NN graph over corpus embeddings.

        Args:
            node_ids        : list of all corpus node IDs.
            node_embeddings : (N, d) embedding matrix for all nodes.
            k_neighbors     : number of nearest neighbors per node.

        Returns:
            self (fluent)
        """
        N, d = node_embeddings.shape
        if N == 0:
            return self

        # L2-normalise for cosine similarity.
        norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
        E = node_embeddings / (norms + 1e-9)

        import scipy.sparse as sp
        from sklearn.cluster import MiniBatchKMeans

        top_k = min(k_neighbors, N - 1)
        
        # ── Approximate Nearest Neighbors (IVF approach) ──
        # Exact all-pairs k-NN on 546k documents takes 40 Trillion operations (~15 mins).
        # We approximate it by clustering into 2048 Voronoi cells, and only computing
        # similarities between documents in the SAME cluster.
        
        n_clusters = min(2048, N // 50)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10240, n_init=1, random_state=42)
        labels = kmeans.fit_predict(E)

        row_list = []
        col_list = []
        val_list = []

        # Process each cluster
        for c in range(n_clusters):
            # Indices of documents in this cluster
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                continue
                
            # Compute all-pairs similarity WITHIN the cluster
            # This is extremely fast because len(idx) ~ 250. 
            # 250x250 = 62,500 pairs, instead of 300 billion.
            sub_E = E[idx]
            sims = sub_E @ sub_E.T
            
            # Zero self-similarity
            np.fill_diagonal(sims, -1.0)
            
            # Find top-K within the cluster (or fewer if cluster is small)
            k_c = min(top_k, len(idx) - 1)
            if k_c <= 0:
                continue
                
            # Partition to find top k_c
            topk_sub_idx = np.argpartition(sims, -k_c, axis=1)[:, -k_c:]
            
            # Extract values
            row_offsets = np.arange(len(idx))[:, None]
            topk_sims = sims[row_offsets, topk_sub_idx]
            
            # Map back to global indices
            global_rows = np.broadcast_to(idx[row_offsets], topk_sub_idx.shape)
            global_cols = idx[topk_sub_idx]
            
            # Filter positive edges
            mask = topk_sims > 0.0
            
            row_list.append(global_rows[mask])
            col_list.append(global_cols[mask])
            val_list.append(topk_sims[mask])

        # Concatenate and build sparse matrix
        if len(row_list) > 0:
            row_idx = np.concatenate(row_list)
            col_idx = np.concatenate(col_list)
            data_vals = np.concatenate(val_list)
        else:
            row_idx, col_idx, data_vals = np.array([]), np.array([]), np.array([])

        A = sp.coo_matrix((data_vals, (row_idx, col_idx)), shape=(N, N)).tocsr()

        # Symmetrise
        A = (A + A.T)
        A.data *= 0.5  # average out bidirectional edges

        self._A_global = A
        self._global_node_ids = list(node_ids)
        self._global_node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        self._global_node_embeds = node_embeddings.astype(np.float32)

        # Fit spectral embedding.
        if N > 1:
            self._global_embed.fit(A, node_ids)

        return self

    # ------------------------------------------------------------------
    # Prefetch (per query turn)
    # ------------------------------------------------------------------

    def prefetch(
        self,
        active_node_ids: list,
        active_embeds: np.ndarray,
        retrieval_seed: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple]:
        """Run walk + cognitive filter to select prefetch candidates.

        Args:
            active_node_ids : IDs of nodes in V_active post-Fiedler cut.
            active_embeds   : (|V_active|, d) embeddings of active nodes.
            retrieval_seed  : r_t — seed vector from Phase 3.
            top_k           : desired number of prefetched nodes.

        Returns:
            selected : list of (node_id, score) tuples, ≤ top_k items.
        """
        if (self._A_global is None
                or self._global_node_ids is None
                or len(self._global_node_ids) == 0):
            return []

        N = len(self._global_node_ids)
        if N == 0:
            return []

        # If global graph is tiny or walk is disabled, fall back to cosine.
        if N <= 5 or not self._global_embed.fitted:
            return self._cosine_fallback(retrieval_seed, top_k)

        # 1. Spectrally-biased random walk from active nodes.
        # Map active_node_ids to global IDs (best-effort).
        start_ids = [
            nid for nid in active_node_ids
            if nid in self._global_node_ids
        ]
        if not start_ids:
            # Seed from retrieval vector: find closest node in global graph.
            norms = np.linalg.norm(self._global_node_embeds, axis=1)
            sims = (
                self._global_node_embeds
                / (norms[:, None] + 1e-9)
            ) @ retrieval_seed
            best = int(np.argmax(sims))
            start_ids = [self._global_node_ids[best]]

        pi = self._walker.walk(
            start_ids,
            self._A_global,
            self._global_embed,
            self._global_node_ids,
            self._rng,
        )

        # 2. Top-K candidates by visitation count.
        k_cand = min(self.top_k_candidates, N)
        top_idx = np.argpartition(pi, -k_cand)[-k_cand:]
        top_idx = top_idx[np.argsort(pi[top_idx])[::-1]]

        candidates = [
            (
                self._global_node_ids[i],
                float(pi[i]),
                self._global_node_embeds[i],
            )
            for i in top_idx
            if pi[i] > 0
        ]

        # 3. Cognitive-load-bounded greedy selection.
        # Ensure we pass the global embeddings (136-D) instead of the local hypergraph embeddings (128-D)
        # to prevent shape mismatch in the cognitive filter.
        active_global_indices = [
            self._global_node_id_to_idx[nid] for nid in active_node_ids
            if nid in self._global_node_id_to_idx
        ]
        if active_global_indices:
            global_active_embeds = self._global_node_embeds[active_global_indices]
        else:
            global_active_embeds = np.zeros((1, self._global_node_embeds.shape[1]), dtype=np.float32)

        selected = self._cog_model.greedy_knapsack(candidates, global_active_embeds)

        return selected[:top_k]

    def _cosine_fallback(
        self, retrieval_seed: np.ndarray, top_k: int
    ) -> list[tuple]:
        """Cosine similarity fallback when global graph is unavailable."""
        E = self._global_node_embeds
        if E is None or len(E) == 0:
            return []
        norms = np.linalg.norm(E, axis=1)
        sims = E @ retrieval_seed / (norms + 1e-9)
        k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [
            (self._global_node_ids[i], float(sims[i]))
            for i in top_idx
        ]

    def reset_session(self) -> None:
        """Reset per-session state (re-seed the RNG)."""
        self._rng = np.random.default_rng()   # fresh seed each session
