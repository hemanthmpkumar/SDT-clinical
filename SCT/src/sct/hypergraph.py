"""
Phase 1 — Multi-Modal Hypergraph Construction (SCT)
====================================================
Five vertex modalities: spectral, photometric, imaging, model, literature.
Four hyperedge classes: spectro-physical, cross-modality, temporal, literature-evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp


class ModalityType(str, Enum):
    SPECTRAL = "spectral"
    PHOTOMETRIC = "photometric"
    IMAGING = "imaging"
    MODEL = "model"
    LITERATURE = "literature"


class HyperedgeType(str, Enum):
    SPECTRO_PHYSICAL = "spectro_physical"
    CROSS_MODALITY = "cross_modality"
    TEMPORAL_COHERENCE = "temporal_coherence"
    LITERATURE_EVIDENCE = "literature_evidence"


# Modality degree normalization weights (γ_m in the proposal)
GAMMA = {
    ModalityType.SPECTRAL:     1.0,
    ModalityType.PHOTOMETRIC:  0.9,
    ModalityType.IMAGING:      0.8,
    ModalityType.MODEL:        0.7,
    ModalityType.LITERATURE:   0.5,
}


@dataclass
class Node:
    """A single vertex in the SCT hypergraph."""
    node_id: str
    modality: ModalityType
    embedding: np.ndarray          # shape (d,)
    timestamp: float               # Unix time or synthetic step index
    chi2: Optional[float] = None   # Reduced χ² (model nodes)
    quality_flag: float = 1.0      # 1=clean, 0=contaminated
    metadata: Dict = field(default_factory=dict)

    @property
    def gamma(self) -> float:
        return GAMMA[self.modality]


@dataclass
class Hyperedge:
    """A single hyperedge connecting ≥ 2 nodes."""
    edge_id: str
    node_ids: Set[str]
    weight: float
    edge_type: HyperedgeType


class MultiModalHypergraph:
    """
    Dynamic multi-modal hypergraph for SCT.

    Maintains vertices and hyperedges in a sliding window of size K.
    Provides incidence-matrix construction for Laplacian computation.

    Parameters
    ----------
    window_size : int
        Maximum number of active nodes (K in proposal).
    tau_min : float
        Minimum hyperedge weight to form an edge.
    tau_sem : float
        Minimum cosine similarity for cross-modality semantic edges.
    sigma_t : float
        Temporal coherence decay timescale (days or steps).
    """

    def __init__(
        self,
        window_size: int = 50,
        tau_min: float = 0.10,
        tau_sem: float = 0.35,
        sigma_t: float = 30.0,
        beta_lit: float = 0.5,
        tau_rec: float = 2.0,
    ):
        self.window_size = window_size
        self.tau_min = tau_min
        self.tau_sem = tau_sem
        self.sigma_t = sigma_t
        self.beta_lit = beta_lit
        self.tau_rec = tau_rec

        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Hyperedge] = {}
        self._step: int = 0
        self._node_order: List[str] = []  # insertion order for window management

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> List[Hyperedge]:
        """
        Add a new vertex; evaluate and add admissible hyperedges.
        Returns list of newly created hyperedges.
        """
        self._nodes[node.node_id] = node
        self._node_order.append(node.node_id)
        self._step += 1

        # Form new hyperedges with existing active nodes
        new_edges = self._evaluate_new_edges(node)
        for edge in new_edges:
            self._edges[edge.edge_id] = edge

        # Expire oldest nodes outside window
        self._expire_old_nodes()
        return new_edges

    @property
    def nodes(self) -> Dict[str, Node]:
        return self._nodes

    @property
    def edges(self) -> Dict[str, Hyperedge]:
        return self._edges

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    def get_node_list(self) -> List[str]:
        return list(self._nodes.keys())

    def build_laplacian(self) -> Tuple[sp.csr_matrix, List[str]]:
        """
        Build the normalized hypergraph Laplacian Δ = I - D_V^{-1/2} H W D_E^{-1} H^T D_V^{-1/2}
        following Zhou et al. (2006).

        Returns
        -------
        Delta : scipy.sparse.csr_matrix  shape (n, n)
        node_ids : list of node IDs in row/column order
        """
        node_ids = list(self._nodes.keys())
        edge_ids = list(self._edges.keys())
        n = len(node_ids)
        m = len(edge_ids)

        if n == 0 or m == 0:
            return sp.eye(max(n, 1), format="csr"), node_ids

        nid2idx = {nid: i for i, nid in enumerate(node_ids)}
        eid2idx = {eid: j for j, eid in enumerate(edge_ids)}

        # Build incidence matrix H (n x m)
        rows, cols, data = [], [], []
        for eid, edge in self._edges.items():
            j = eid2idx[eid]
            for nid in edge.node_ids:
                if nid in nid2idx:
                    rows.append(nid2idx[nid])
                    cols.append(j)
                    data.append(1.0)
        H = sp.csr_matrix((data, (rows, cols)), shape=(n, m))

        # Vertex modality weights
        gamma_vals = np.array([self._nodes[nid].gamma for nid in node_ids], dtype=float)
        H_w = sp.diags(gamma_vals) @ H

        # Edge degree D_E = diag(sum_v H_w[v, e])  (m x m)
        delta_e = np.maximum(np.array(H_w.sum(axis=0)).flatten(), 1.0)
        D_E_inv = sp.diags(1.0 / delta_e)

        # Edge weight diagonal W (m x m)
        w_vals = np.array([self._edges[eid].weight for eid in edge_ids], dtype=float)
        W = sp.diags(w_vals)

        # Hypergraph affinity matrix A = H_w W D_E^{-1} H_w^T
        A = H_w @ W @ D_E_inv @ H_w.T

        # Vertex degree d_v = sum_u A[v, u]
        d_v = np.maximum(np.array(A.sum(axis=1)).flatten(), 1e-10)
        D_V_inv_sqrt = sp.diags(1.0 / np.sqrt(d_v))

        # Normalized hypergraph Laplacian Delta = I - D_V^{-1/2} A D_V^{-1/2}
        Theta = D_V_inv_sqrt @ A @ D_V_inv_sqrt
        Delta = sp.eye(n, format="csr") - Theta.tocsr()
        return Delta, node_ids

    def get_node_embeddings_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return (n, d) embedding matrix and ordered node IDs."""
        node_ids = list(self._nodes.keys())
        if not node_ids:
            return np.empty((0, 1)), node_ids
        embs = np.stack([self._nodes[nid].embedding for nid in node_ids])
        return embs, node_ids

    def remove_nodes(self, node_ids: Set[str]):
        """Remove a set of nodes and all hyperedges referencing them."""
        for nid in node_ids:
            self._nodes.pop(nid, None)
            if nid in self._node_order:
                self._node_order.remove(nid)

        # Remove edges that now have < 2 valid members
        dead_edges = []
        for eid, edge in self._edges.items():
            edge.node_ids -= node_ids
            if len(edge.node_ids) < 2:
                dead_edges.append(eid)
        for eid in dead_edges:
            self._edges.pop(eid)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_new_edges(self, new_node: Node) -> List[Hyperedge]:
        """Evaluate all admissible hyperedge types for new_node vs. existing nodes."""
        new_edges: List[Hyperedge] = []
        existing = [(nid, n) for nid, n in self._nodes.items() if nid != new_node.node_id]

        for nid, existing_node in existing:
            # (a) Cross-modality semantic edge (different modalities)
            if existing_node.modality != new_node.modality:
                w = self._cross_modality_weight(new_node, existing_node)
                if w >= self.tau_min:
                    eid = f"CM_{new_node.node_id}__{nid}"
                    new_edges.append(Hyperedge(
                        edge_id=eid,
                        node_ids={new_node.node_id, nid},
                        weight=w,
                        edge_type=HyperedgeType.CROSS_MODALITY,
                    ))

            # (b) Temporal coherence edge (any modality, same target)
            w_tc = self._temporal_weight(new_node, existing_node)
            if w_tc >= self.tau_min:
                eid = f"TC_{new_node.node_id}__{nid}"
                new_edges.append(Hyperedge(
                    edge_id=eid,
                    node_ids={new_node.node_id, nid},
                    weight=w_tc,
                    edge_type=HyperedgeType.TEMPORAL_COHERENCE,
                ))

            # (c) Spectro-physical consistency (spectral ↔ model)
            if (new_node.modality == ModalityType.SPECTRAL and
                    existing_node.modality == ModalityType.MODEL):
                w_sp = self._spectro_physical_weight(new_node, existing_node)
                if w_sp >= self.tau_min:
                    eid = f"SP_{new_node.node_id}__{nid}"
                    new_edges.append(Hyperedge(
                        edge_id=eid,
                        node_ids={new_node.node_id, nid},
                        weight=w_sp,
                        edge_type=HyperedgeType.SPECTRO_PHYSICAL,
                    ))
            elif (new_node.modality == ModalityType.MODEL and
                  existing_node.modality == ModalityType.SPECTRAL):
                w_sp = self._spectro_physical_weight(existing_node, new_node)
                if w_sp >= self.tau_min:
                    eid = f"SP_{nid}__{new_node.node_id}"
                    new_edges.append(Hyperedge(
                        edge_id=eid,
                        node_ids={new_node.node_id, nid},
                        weight=w_sp,
                        edge_type=HyperedgeType.SPECTRO_PHYSICAL,
                    ))

            # (d) Literature-evidence edge
            if (new_node.modality == ModalityType.LITERATURE or
                    existing_node.modality == ModalityType.LITERATURE):
                w_le = self._literature_evidence_weight(new_node, existing_node)
                if w_le >= self.tau_min:
                    eid = f"LE_{new_node.node_id}__{nid}"
                    new_edges.append(Hyperedge(
                        edge_id=eid,
                        node_ids={new_node.node_id, nid},
                        weight=w_le,
                        edge_type=HyperedgeType.LITERATURE_EVIDENCE,
                    ))

        return new_edges

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _cross_modality_weight(self, u: Node, v: Node) -> float:
        sim = self._cosine_sim(u.embedding, v.embedding)
        return sim if sim >= self.tau_sem else 0.0

    def _temporal_weight(self, u: Node, v: Node) -> float:
        dt = abs(u.timestamp - v.timestamp)
        return float(np.exp(-dt / self.sigma_t))

    def _spectro_physical_weight(self, spec_node: Node, model_node: Node) -> float:
        """
        Weight = exp(-chi2) * cosine_sim, clamped by quality flag.
        """
        chi2 = model_node.chi2 if model_node.chi2 is not None else 2.0
        cos_sim = self._cosine_sim(spec_node.embedding, model_node.embedding)
        w = np.exp(-chi2) * cos_sim
        w *= spec_node.quality_flag * model_node.quality_flag
        return float(max(w, 0.0))

    def _literature_evidence_weight(self, u: Node, v: Node) -> float:
        """Semantic similarity weighted by recency."""
        sim = self._cosine_sim(u.embedding, v.embedding)
        t_now = time.time()
        # Use metadata 'pub_time' if available
        pub_u = u.metadata.get("pub_time", t_now)
        pub_v = v.metadata.get("pub_time", t_now)
        recency = np.exp(-abs(t_now - min(pub_u, pub_v)) / (self.tau_rec * 3.15e7))
        return float(sim * (self.beta_lit + (1 - self.beta_lit) * recency))

    def _expire_old_nodes(self):
        while len(self._node_order) > self.window_size:
            oldest_id = self._node_order.pop(0)
            if oldest_id in self._nodes:
                del self._nodes[oldest_id]
            # Clean up orphaned edges
            dead = [eid for eid, e in self._edges.items() if oldest_id in e.node_ids]
            for eid in dead:
                del self._edges[eid]
