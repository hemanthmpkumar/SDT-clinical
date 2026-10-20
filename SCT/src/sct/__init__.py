"""
SCT Package — Spectral Cosmic Trajectories
==========================================
A hypergraph-spectral framework for adaptive scientific discovery
in planetary and astrophysical data streams.
"""
from .hypergraph import MultiModalHypergraph, Node, HyperedgeType
from .fiedler import ScientificPivotDetector
from .attenuation import ChebyshevFiedlerCut
from .prefetch import AdaptiveObservationPrefetcher
from .metrics import SCTMetrics

__version__ = "1.0.0"
__all__ = [
    "MultiModalHypergraph",
    "Node",
    "HyperedgeType",
    "ScientificPivotDetector",
    "ChebyshevFiedlerCut",
    "AdaptiveObservationPrefetcher",
    "SCTMetrics",
]
