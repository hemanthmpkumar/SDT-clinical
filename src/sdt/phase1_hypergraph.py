#!/usr/bin/env python3
"""
src/sdt/phase1_hypergraph.py

Phase 1 — Multi-Modal Hypergraph State Space
=============================================

Implements the dynamic hypergraph G_t = (V_t, E_t, W_t) that replaces
GDT's SPD matrix S_t as the instantaneous diagnostic state representation.

Three node modalities co-exist in the same graph:
  • Text / semantic nodes  — TF-IDF → learned linear projection to R^d
  • Structured EHR nodes   — ICD / lab / medication events (same encoder)
  • Physiological nodes    — 1D-CNN sliding-window over waveform segments
                             (simulated when real MIMIC-IV waveforms unavailable)

Hyperedges connect arbitrary subsets of nodes (not just pairs), weighted by:
    w(v_i, v_j) = α · w_seq(v_i, v_j)  +  (1-α) · w_sem(v_i, v_j)

where
    w_seq = exp(-|t_i - t_j| / σ_seq)   temporal sequential proximity
    w_sem = cosine(ê_i, ê_j)             semantic cosine similarity

The adjacency matrix follows Zhou et al. (2006):
    A_t = H_t W_t B^{-1} H_t^T

References
----------
Zhou et al. (2006) "Learning with Hypergraphs" — NeurIPS.
Feng et al. (2019) "Hypergraph Neural Networks" — AAAI.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Lightweight 1D-CNN waveform encoder (no PyTorch dependency in this module).
# In production, swap PhysioEncoder with a real Mamba/S4 encoder loaded from
# a checkpoint.
# ---------------------------------------------------------------------------

class PhysioEncoder:
    """Simulated physiological waveform encoder (1D-CNN surrogate).

    Takes a raw waveform window of shape (window_len,) and returns a
    d-dimensional embedding via a fixed random projection followed by a
    non-linearity — sufficient for benchmarking the graph topology
    without real waveform data.

    To use real waveforms, subclass and override `encode(segment)` with
    your Mamba / S4 encoder.
    """

    def __init__(self, window_len: int = 256, d: int = 128, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(d, window_len).astype(np.float32) / np.sqrt(window_len)
        self.b = rng.randn(d).astype(np.float32) * 0.01
        self.d = d
        self.window_len = window_len

    def encode(self, segment: np.ndarray) -> np.ndarray:
        """Encode a waveform segment to R^d.

        Args:
            segment: 1-D float array of length window_len.  If shorter,
                     it is zero-padded; if longer, it is truncated.

        Returns:
            L2-normalised embedding vector of shape (d,).
        """
        seg = np.asarray(segment, dtype=np.float32)
        if len(seg) < self.window_len:
            seg = np.pad(seg, (0, self.window_len - len(seg)))
        else:
            seg = seg[: self.window_len]
        h = np.tanh(self.W @ seg + self.b)           # (d,)
        norm = np.linalg.norm(h)
        return h / (norm + 1e-9)

    def simulate(self, seed: int = None) -> np.ndarray:
        """Generate a synthetic waveform embedding for testing."""
        rng = np.random.RandomState(seed)
        return self.encode(rng.randn(self.window_len).astype(np.float32))


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

class Node:
    """A single node in the multi-modal hypergraph."""

    __slots__ = ("node_id", "modality", "embedding", "timestamp", "metadata")

    MODALITY_TEXT   = "text"
    MODALITY_PHYSIO = "physio"
    MODALITY_EHR    = "ehr"

    def __init__(
        self,
        node_id: str,
        modality: str,
        embedding: np.ndarray,
        timestamp: float = 0.0,
        metadata: Optional[dict] = None,
    ):
        self.node_id   = node_id
        self.modality  = modality
        self.embedding = embedding.astype(np.float32)
        self.timestamp = float(timestamp)
        self.metadata  = metadata or {}


# ---------------------------------------------------------------------------
# MultiModalHypergraph — Phase 1 core
# ---------------------------------------------------------------------------

class MultiModalHypergraph:
    """Dynamic multi-modal hypergraph G_t = (V_t, E_t, W_t).

    Parameters
    ----------
    d : int
        Projected embedding dimension (default 128, per paper §3).
    alpha : float
        Balance between temporal (w_seq) and semantic (w_sem) edge weights.
        α=1 → purely temporal; α=0 → purely semantic.
    sigma_seq : float
        Decay constant for temporal weight w_seq (in turn units).
    window_k : int
        Sliding window size: only nodes from the last k turns are retained.
    sem_threshold : float
        Minimum cosine similarity to form a hyperedge.
    physio_window_len : int
        Waveform segment length fed to PhysioEncoder.
    """

    def __init__(
        self,
        d: int = 128,
        alpha: float = 0.5,
        sigma_seq: float = 3.0,
        window_k: int = 8,
        sem_threshold: float = 0.10,
        physio_window_len: int = 256,
        seed: int = 42,
    ):
        self.d = d
        self.alpha = alpha
        self.sigma_seq = sigma_seq
        self.window_k = window_k
        self.sem_threshold = sem_threshold
        self.seed = seed

        # Fitted TF-IDF vectorizer + linear projection W_proj ∈ R^{d × vocab}
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._W_proj: Optional[np.ndarray] = None  # (d, vocab)

        # Physiological encoder
        self._physio_enc = PhysioEncoder(
            window_len=physio_window_len, d=d, seed=seed
        )

        # Active node list (rolling window)
        self._nodes: list[Node] = []
        self._turn: int = 0  # current turn index

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, corpus_texts: list[str]) -> "MultiModalHypergraph":
        """Fit TF-IDF vocabulary and learn the linear projection W_proj.

        Args:
            corpus_texts: list of document strings from the corpus.

        Returns:
            self (fluent API)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._vectorizer = TfidfVectorizer(
            max_df=0.85, min_df=1, stop_words="english",
            max_features=4000, sublinear_tf=True,
        )
        X = self._vectorizer.fit_transform(corpus_texts)  # (N, vocab)
        vocab = X.shape[1]

        # Learn a d-dimensional projection by truncated SVD on the corpus.
        n_components = min(self.d, vocab - 1, X.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        svd.fit(X)
        # W_proj shape: (n_components, vocab) → transpose to (d, vocab)
        W = svd.components_           # (n_components, vocab)
        if n_components < self.d:
            pad = np.zeros((self.d - n_components, vocab), dtype=np.float32)
            W = np.vstack([W, pad])
        self._W_proj = W.astype(np.float32)   # (d, vocab)
        return self

    # ------------------------------------------------------------------
    # Node creation
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> np.ndarray:
        """TF-IDF → linear projection → L2-normalise → R^d."""
        if self._vectorizer is None or self._W_proj is None:
            raise RuntimeError("Call fit() before adding text nodes.")
        tfidf = self._vectorizer.transform([text])            # (1, vocab) sparse
        res = self._W_proj @ tfidf.T                          # (d, 1)
        if hasattr(res, "toarray"):
            vec = res.toarray().ravel()
        else:
            vec = np.asarray(res).ravel()
        norm = np.linalg.norm(vec)
        return (vec / (norm + 1e-9)).astype(np.float32)

    def add_text_node(
        self,
        node_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> Node:
        """Create and register a text/semantic node from a clinical query or note."""
        emb = self._encode_text(text)
        node = Node(
            node_id=node_id,
            modality=Node.MODALITY_TEXT,
            embedding=emb,
            timestamp=float(self._turn),
            metadata=metadata or {},
        )
        self._nodes.append(node)
        self._trim_window()
        return node

    def add_ehr_node(
        self,
        node_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> Node:
        """Create and register a structured EHR event node (ICD, lab, medication)."""
        emb = self._encode_text(text)
        node = Node(
            node_id=node_id,
            modality=Node.MODALITY_EHR,
            embedding=emb,
            timestamp=float(self._turn),
            metadata=metadata or {},
        )
        self._nodes.append(node)
        self._trim_window()
        return node

    def add_physio_node(
        self,
        node_id: str,
        waveform: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
    ) -> Node:
        """Create a physiological node from a waveform segment.

        If waveform is None, generates a simulated embedding (for testing).
        """
        if waveform is not None:
            emb = self._physio_enc.encode(waveform)
        else:
            emb = self._physio_enc.simulate(seed=hash(node_id) % (2**31))
        node = Node(
            node_id=node_id,
            modality=Node.MODALITY_PHYSIO,
            embedding=emb,
            timestamp=float(self._turn),
            metadata=metadata or {},
        )
        self._nodes.append(node)
        self._trim_window()
        return node

    def advance_turn(self) -> None:
        """Increment the internal turn counter (call after each clinician query)."""
        self._turn += 1

    def _trim_window(self) -> None:
        """Keep only the last window_k turns in the active node list."""
        if len(self._nodes) > 0:
            cutoff = self._turn - self.window_k
            self._nodes = [n for n in self._nodes if n.timestamp > cutoff]

    # ------------------------------------------------------------------
    # Hyperedge formation and adjacency construction
    # ------------------------------------------------------------------

    def _w_seq(self, t_i: float, t_j: float) -> float:
        """Temporal sequential weight (Eq. 4 from paper)."""
        return float(np.exp(-abs(t_i - t_j) / self.sigma_seq))

    def _w_sem(self, e_i: np.ndarray, e_j: np.ndarray) -> float:
        """Semantic cosine weight (Eq. 5 from paper)."""
        ni = np.linalg.norm(e_i)
        nj = np.linalg.norm(e_j)
        if ni < 1e-9 or nj < 1e-9:
            return 0.0
        return float(np.dot(e_i, e_j) / (ni * nj))

    def _composite_weight(self, n_i: Node, n_j: Node) -> float:
        """Composite edge weight (Eq. 6 from paper):
           w(v_i, v_j) = α·w_seq + (1-α)·w_sem
        """
        ws = self._w_seq(n_i.timestamp, n_j.timestamp)
        wc = self._w_sem(n_i.embedding, n_j.embedding)
        return self.alpha * ws + (1.0 - self.alpha) * wc

    def build_adjacency(self) -> tuple[np.ndarray, list[str]]:
        """Build the hypergraph adjacency matrix A_t = H W B^{-1} H^T.

        Only node pairs with composite weight ≥ sem_threshold form a hyperedge.

        Returns
        -------
        A : np.ndarray, shape (n, n)
            Weighted adjacency matrix of the active subgraph.
        node_ids : list[str]
            Node IDs in the same row/column order as A.
        """
        nodes = self._nodes
        n = len(nodes)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32), []

        node_ids = [nd.node_id for nd in nodes]

        # Build hyperedges: each ordered pair (i,j) with w > threshold.
        # For small subgraphs this is equivalent to a weighted pairwise graph,
        # which still satisfies the hypergraph adjacency formula.
        hyperedges: list[tuple[int, int, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                w = self._composite_weight(nodes[i], nodes[j])
                if w >= self.sem_threshold:
                    hyperedges.append((i, j, w))

        if not hyperedges:
            # All nodes isolated — return diagonal identity.
            return np.eye(n, dtype=np.float32), node_ids

        # Build incidence matrix H ∈ {0,1}^{n × |E|}
        # and edge weight diagonal W_e ∈ R^{|E|}
        n_edges = len(hyperedges)
        H = np.zeros((n, n_edges), dtype=np.float32)
        W_e = np.zeros(n_edges, dtype=np.float32)
        for e_idx, (i, j, w) in enumerate(hyperedges):
            H[i, e_idx] = 1.0
            H[j, e_idx] = 1.0
            W_e[e_idx] = w

        # Vertex degree: B = diag(H W_e 1)
        B_diag = H @ W_e                          # (n,)
        B_inv = np.where(B_diag > 1e-9, 1.0 / B_diag, 0.0)

        # A_t = H W_e B^{-1} H^T   (Zhou et al. 2006, Eq. adjacency)
        A = (H * W_e[np.newaxis, :]) @ (H * B_inv[:, np.newaxis]).T
        # Symmetrise numerically and add tiny ridge for numerical stability.
        A = 0.5 * (A + A.T) + 1e-6 * np.eye(n, dtype=np.float32)

        return A.astype(np.float32), node_ids

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def active_nodes(self) -> list[Node]:
        """Current active node list (last window_k turns)."""
        return list(self._nodes)

    @property
    def n_active(self) -> int:
        return len(self._nodes)

    def embedding_matrix(self) -> np.ndarray:
        """Stack active node embeddings into shape (n_active, d)."""
        if not self._nodes:
            return np.zeros((0, self.d), dtype=np.float32)
        return np.stack([nd.embedding for nd in self._nodes], axis=0)

    def reset(self) -> None:
        """Clear per-session state (call between patients)."""
        self._nodes.clear()
        self._turn = 0
