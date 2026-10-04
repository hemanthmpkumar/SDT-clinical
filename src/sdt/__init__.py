"""
src/sdt/__init__.py

Spectral Diagnostic Trajectories (SDT) package.

Public API
----------
    from src.sdt import SDTRetriever

    retriever = SDTRetriever(corpus, tau_pivot=0.15, alpha=0.5)
    retriever.fit_predictor(vignettes)
    results = retriever.search(query, session_history)
"""

from .sdt_retriever import SDTRetriever
from .phase1_hypergraph import MultiModalHypergraph
from .phase2_fiedler import SpectralPivotDetector
from .phase3_attenuation import GraphSignalAttenuator
from .phase4_prefetch import AnticipatoryPrefetchEngine
from .phase5_evaluation import SDTEvaluator, erlang_c, compute_wq, ed_impact_table

__all__ = [
    "SDTRetriever",
    "MultiModalHypergraph",
    "SpectralPivotDetector",
    "GraphSignalAttenuator",
    "AnticipatoryPrefetchEngine",
    "SDTEvaluator",
    "erlang_c",
    "compute_wq",
    "ed_impact_table",
]
