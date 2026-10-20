#!/usr/bin/env python3
"""
src/sdt/sdt_retriever.py

SDTRetriever — Top-level SDT orchestrator
==========================================

Implements the full Spectral Diagnostic Trajectories pipeline across all
five phases and exposes the same retriever interface as GDTRetriever, so
it can be dropped directly into the existing experimental harness
(src/experiments/run.py / simulate_users.py) as a fifth arm.

Pipeline per query turn
-----------------------
  1. [Phase 1]  Add a text node for the new query; optionally add physio nodes.
  2. [Phase 1]  Build hypergraph adjacency A_t from the active window.
  3. [Phase 2]  Compute λ₂(L_t); check interference gate (Δλ₂ > τ_pivot).
  4. [Phase 3]  If gate fired: apply Fiedler cut → filtered retrieval seed r_t.
                Otherwise:      use the plain mean embedding as seed.
  5. [Phase 4]  Spectrally-biased random walk → prefetch candidates → cognitive filter.
  6.  Score corpus documents: BM25-style TF-IDF  +  cosine(r_t, doc)  +  prefetch boost.
  7.  Return top-K ranked (note_id, score) pairs.

Parameters
----------
tau_pivot : float
    Fiedler gate threshold (Eq. 14).
alpha : float
    Temporal vs. semantic balance in hyperedge weights.
window_k : int
    Sliding window size for active subgraph.
sem_threshold : float
    Minimum edge weight to form a hyperedge.
gamma_filter : float
    High-pass filter sharpness (Phase 3).
spectral_m : int
    Global spectral embedding dimension (Phase 4).
tau_walk : float
    Spectral bias strength in random walk.
n_walk : int
    Walk length per node.
c_max : float
    Cognitive load budget (Sweller bound).
prefetch_weight : float
    Weight of prefetch score component in final ranking.
semantic_weight : float
    Weight of cosine(r_t, doc) in final ranking.
lexical_weight : float
    Weight of TF-IDF lexical score in final ranking.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from src.models.base import BaseRetriever, build_tfidf
from src.models.cma import CMARetriever

from .phase1_hypergraph import MultiModalHypergraph, Node
from .phase2_fiedler import SpectralPivotDetector
from .phase3_attenuation import GraphSignalAttenuator
from .phase4_prefetch import AnticipatoryPrefetchEngine
from .phase5_evaluation import SDTEvaluator, QueryResult

warnings.filterwarnings("ignore", category=RuntimeWarning)


class SDTRetriever(BaseRetriever):
    """Spectral Diagnostic Trajectories (SDT) retriever.

    Extends BaseRetriever with the full SDT five-phase pipeline.
    """

    def __init__(
        self,
        corpus: list[dict],
        # ── Phase 2 ──────────────────────────────
        tau_pivot: float = 0.15,
        # ── Phase 1 ──────────────────────────────
        alpha: float = 0.5,
        window_k: int = 8,
        sem_threshold: float = 0.10,
        hypergraph_d: int = 128,
        sigma_seq: float = 3.0,
        # ── Phase 3 ──────────────────────────────
        gamma_filter: float = 3.0,
        poly_order: int = 5,
        # ── Phase 4 ──────────────────────────────
        spectral_m: int = 16,
        tau_walk: float = 2.0,
        n_walk: int = 30,
        n_restarts: int = 3,
        c_max: float = 7.0,
        top_k_candidates: int = 50,
        # ── Scoring weights ───────────────────────
        lexical_weight: float = 1.0,
        semantic_weight: float = 0.20,
        prefetch_weight: float = 0.10,
        # ── Shared encoder (from CMA/GDT, optional) ─
        cma_retriever=None,
        # ── Misc ─────────────────────────────────
        seed: int = 42,
    ):
        super().__init__(corpus)

        # ── Reuse CMA/GDT encoder if provided, else build fresh ──────────
        doc_texts = [rec["text"] for rec in corpus]
        if cma_retriever is not None:
            self._vectorizer = cma_retriever.vectorizer
            self._doc_tfidf  = cma_retriever.doc_tfidf
            self._doc_latent = cma_retriever.doc_latent   # (N, latent_dim)
            self._cma_encoder = getattr(cma_retriever, "encoder", None)
        else:
            self._vectorizer = build_tfidf(doc_texts)
            self._doc_tfidf  = self._vectorizer.transform(doc_texts)
            self._doc_latent = None   # will use TF-IDF cosine only
            self._cma_encoder = None

        self.lexical_weight  = lexical_weight
        self.semantic_weight = semantic_weight
        self.prefetch_weight = prefetch_weight

        # ── Phase 1: Hypergraph ───────────────────────────────────────────
        self._hypergraph = MultiModalHypergraph(
            d=hypergraph_d,
            alpha=alpha,
            sigma_seq=sigma_seq,
            window_k=window_k,
            sem_threshold=sem_threshold,
            seed=seed,
        )
        self._hypergraph.fit(doc_texts)

        # ── Phase 2: Spectral pivot detector ─────────────────────────────
        self._pivot_detector = SpectralPivotDetector(
            tau_pivot=tau_pivot,
            min_nodes=3,
        )

        # ── Phase 3: Graph signal attenuator ─────────────────────────────
        self._attenuator = GraphSignalAttenuator(
            gamma=gamma_filter,
            poly_order=poly_order,
        )

        # ── Phase 4: Prefetch engine ──────────────────────────────────────
        self._prefetch_engine = AnticipatoryPrefetchEngine(
            m=spectral_m,
            tau=tau_walk,
            n_walk=n_walk,
            n_restarts=n_restarts,
            c_max=c_max,
            top_k_candidates=top_k_candidates,
            seed=seed,
        )

        # ── Phase 5: Evaluator ────────────────────────────────────────────
        self.evaluator = SDTEvaluator()

        # ── Per-session state ─────────────────────────────────────────────
        self._session_history: list[str] = []
        self._session_seed:    np.ndarray | None = None   # r_t from Phase 3
        self._gate_triggers:   int = 0
        self._turn:            int = 0

        # ── Build global graph (offline) ──────────────────────────────────
        self._build_global_graph()

    # ------------------------------------------------------------------
    # Global graph initialisation
    # ------------------------------------------------------------------

    def _build_global_graph(self) -> None:
        """Build G_global from corpus doc latents or TF-IDF embeddings."""
        N = len(self.corpus)
        if N < 2:
            return

        # Use doc_latent if available (from shared CMA encoder), else TF-IDF SVD.
        if self._doc_latent is not None and len(self._doc_latent) == N:
            embeddings = self._doc_latent.astype(np.float32)
        else:
            from sklearn.decomposition import TruncatedSVD
            X = self._doc_tfidf
            k = min(64, X.shape[1] - 1, N - 1)
            svd = TruncatedSVD(n_components=k, random_state=42)
            embeddings = svd.fit_transform(X).astype(np.float32)

        node_ids = [rec["note_id"] for rec in self.corpus]
        self._prefetch_engine.build_global_graph(
            node_ids=node_ids,
            node_embeddings=embeddings,
            k_neighbors=min(10, N - 1),
        )

    # ------------------------------------------------------------------
    # Internal encoding helpers
    # ------------------------------------------------------------------

    def _encode_query(self, query: str) -> np.ndarray:
        """TF-IDF → normalised dense vector for semantic scoring."""
        q_tfidf = self._vectorizer.transform([query])
        if self._cma_encoder is not None:
            # Actually encode to the 136-D log-SPD latent space!
            import torch
            import scipy.sparse as sp
            # Extract sparse data and pass to encoder correctly
            if sp.issparse(q_tfidf):
                q_t = torch.tensor(q_tfidf.toarray(), dtype=torch.float32, device=self._cma_encoder.device)
            else:
                q_t = torch.tensor(q_tfidf, dtype=torch.float32, device=self._cma_encoder.device)
            return self._cma_encoder.encode_to_log_vec(q_t)[0].cpu().numpy()
            
        if self._doc_latent is not None:
            # Project using the first latent_dim SVD directions.
            # Quick: we just dot with doc_latent mean as a cosine proxy.
            q_dense = np.asarray(q_tfidf.todense(), dtype=np.float32).ravel()
            norm = np.linalg.norm(q_dense)
            return q_dense / (norm + 1e-9)

        # Plain TF-IDF cosine.
        q_dense = np.asarray(q_tfidf.todense(), dtype=np.float32).ravel()
        norm = np.linalg.norm(q_dense)
        return q_dense / (norm + 1e-9)

    def _lexical_scores(self, query: str, history: list[str]) -> np.ndarray:
        """TF-IDF dot product scores for the expanded query."""
        expanded = " ".join([query] + history[-4:])
        q_tfidf = self._vectorizer.transform([expanded])
        scores = (self._doc_tfidf @ q_tfidf.T).toarray().ravel()
        # Z-score normalise.
        mu, sigma = scores.mean(), scores.std()
        if sigma < 1e-12:
            return scores - mu
        return (scores - mu) / sigma

    # ------------------------------------------------------------------
    # Main retrieval interface  (matches BaseRetriever / GDTRetriever)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        session_history: list[str],
        top_k: int = 10,
        prefetch: bool = True,
        filter_ids: Optional[set] = None,
        **kwargs,
    ) -> list[tuple[str, float]]:
        """Run one turn of the SDT pipeline and return ranked documents.

        Args:
            query           : Current clinician query string.
            session_history : Previous queries in this session.
            top_k           : Number of documents to return.
            prefetch        : Whether to run the Phase 4 prefetch engine.
            filter_ids      : Optional set of note_ids to restrict search to.

        Returns:
            list of (note_id, score) tuples, length ≤ top_k.
        """
        self._turn += 1
        node_id = f"q{self._turn}"

        # ── Phase 1: Add query node to hypergraph ─────────────────────────
        self._hypergraph.add_text_node(node_id, query)
        self._hypergraph.advance_turn()
        A, active_node_ids = self._hypergraph.build_adjacency()
        embeddings = self._hypergraph.embedding_matrix()   # (n, d)
        n = len(active_node_ids)

        # ── Phase 2: Spectral pivot detection ─────────────────────────────
        gate_fired = False
        fiedler_v  = None
        lambda2    = 0.0
        if n >= 2:
            gate_fired, lambda2, fiedler_v = self._pivot_detector.update(A)
            if gate_fired:
                self._gate_triggers += 1

        # ── Phase 3: Context attenuation (only on gate fire) ──────────────
        if gate_fired and fiedler_v is not None and n >= 2:
            from .phase2_fiedler import SpectralPivotDetector as _SPD
            L = _SPD.normalized_laplacian(A)
            node_idx = list(range(n))
            r_t, stale_ids, active_ids, ncut = self._attenuator.attenuate(
                A=A, L=L,
                fiedler_vector=fiedler_v,
                embeddings=embeddings,
                node_idx=node_idx,
                lambda2=lambda2,
            )
            self._session_seed = r_t
            active_local = active_ids
        else:
            # No gate: seed is the mean embedding of the active window.
            if n > 0:
                mean_e = embeddings.mean(axis=0)
                norm = np.linalg.norm(mean_e)
                self._session_seed = mean_e / (norm + 1e-9)
            active_local = list(range(n))

        # ── Lexical scoring ───────────────────────────────────────────────
        scores = self.lexical_weight * self._lexical_scores(query, session_history)

        # ── Semantic scoring: cosine(r_t, doc_tfidf) ─────────────────────
        if self.semantic_weight > 0 and self._session_seed is not None:
            seed = self._session_seed
            # Compute cosine against TF-IDF doc matrix.
            seed_sparse_proxy = np.zeros(
                self._doc_tfidf.shape[1], dtype=np.float32
            )
            # Map hypergraph seed into TF-IDF space via vocabulary alignment.
            # (When doc_latent is available, use it directly.)
            if self._doc_latent is not None:
                q_latent = self._encode_query(query)
                # Compute true cosine similarity against the 136-D neural doc latents
                # instead of padding and doing a meaningless dot product with TF-IDF vocab.
                norms = np.linalg.norm(self._doc_latent, axis=1)
                sem_scores = self._doc_latent @ q_latent / (norms + 1e-9)
                sem_scores = np.asarray(sem_scores).ravel()
            else:
                q_vec = self._encode_query(query)
                sem_scores = self._doc_tfidf @ q_vec
                sem_scores = np.asarray(sem_scores).ravel()
            # Normalise and add.
            s_mu, s_std = sem_scores.mean(), sem_scores.std()
            if s_std > 1e-12:
                sem_scores = (sem_scores - s_mu) / s_std
            scores = scores + self.semantic_weight * sem_scores

        # ── Phase 4: Prefetch boost ───────────────────────────────────────
        if prefetch and self.prefetch_weight > 0 and n >= 2:
            active_embeds = embeddings[active_local] if active_local else embeddings
            seed_vec = self._session_seed if self._session_seed is not None \
                else embeddings.mean(axis=0)

            # Map the seed to the global graph's dimensionality.
            # If we used CMA for the global graph (136-D), the hypergraph local seed (128-D)
            # will crash. We must encode the current query to the CMA latent space.
            if self._doc_latent is not None:
                pf_seed = self._encode_query(query)
            else:
                pf_seed = seed_vec

            prefetched = self._prefetch_engine.prefetch(
                active_node_ids=[active_node_ids[i] for i in active_local],
                active_embeds=active_embeds,
                retrieval_seed=pf_seed,
                top_k=top_k,
            )
            # Boost prefetched doc IDs.
            note_id_set = set(self.note_ids)
            prefetched_ids = {nid for nid, _ in prefetched if nid in note_id_set}
            if prefetched_ids:
                note_id_arr = np.array(self.note_ids)
                pfmask = np.isin(note_id_arr, list(prefetched_ids))
                scores[pfmask] += self.prefetch_weight

        # ── Patient-level filter ──────────────────────────────────────────
        if filter_ids is not None:
            note_arr = np.array(self.note_ids)
            mask = ~np.isin(note_arr, list(filter_ids))
            scores[mask] = -np.inf

        # ── Rank and return ───────────────────────────────────────────────
        ranked = np.argsort(scores)[::-1]
        self._session_history.append(query)

        return [(self.note_ids[i], float(scores[i])) for i in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Predictor fitting (mirrors GDT interface for harness compatibility)
    # ------------------------------------------------------------------

    def fit_predictor(
        self,
        vignettes: list[dict],
        epochs: int = 0,
        batch_size: int = 64,
    ) -> "SDTRetriever":
        """No-op: SDT prefetching is self-supervised via random walks.

        Kept for interface parity with GDT / CMA.
        """
        return self

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Clear all per-session state between patients."""
        self._hypergraph.reset()
        self._pivot_detector.reset()
        self._attenuator.reset()
        self._prefetch_engine.reset_session()
        self.evaluator.reset()
        self._session_history.clear()
        self._session_seed = None
        self._gate_triggers = 0
        self._turn = 0

    def copy_with_hyperparams(self, **overrides) -> "SDTRetriever":
        """Return a new SDTRetriever sharing the fitted corpus index."""
        defaults = dict(
            corpus=self.corpus,
            tau_pivot=self._pivot_detector.tau_pivot,
            lexical_weight=self.lexical_weight,
            semantic_weight=self.semantic_weight,
            prefetch_weight=self.prefetch_weight,
            seed=42,
        )
        defaults.update(overrides)
        return SDTRetriever(**defaults)

    # ------------------------------------------------------------------
    # Diagnostic properties
    # ------------------------------------------------------------------

    @property
    def gate_triggers(self) -> int:
        """Total number of Fiedler gate fires in the current session."""
        return self._gate_triggers

    @property
    def last_lambda2(self) -> float:
        """Most recent Fiedler value λ₂(Lₜ)."""
        return self._pivot_detector.last_lambda2

    @property
    def last_ncut(self) -> float:
        """NCut cost of last Fiedler cut (should be ≤ λ₂)."""
        return self._attenuator.last_ncut

    def spectral_report(self) -> dict:
        """Return a structured diagnostic snapshot of the current state."""
        return {
            **self._pivot_detector.algebraic_connectivity_report(),
            "n_active_nodes":  self._hypergraph.n_active,
            "gate_triggers":   self._gate_triggers,
            "last_ncut":       self._attenuator.last_ncut,
            "session_turn":    self._turn,
        }
