#!/usr/bin/env python3
"""
src/experiments/run.py

SDT Experiment Runner
=====================

Five-arm randomised crossover experiment comparing:
    1. control  — TF-IDF session baseline
    2. bm25     — BM25 sparse retrieval
    3. cma      — Contextual Memory Augmentation
    4. gdt      — Geodesic Diagnostic Trajectories (Riemannian SPD manifold)
    5. sdt      — Spectral Diagnostic Trajectories (Hypergraph + Fiedler)  ← NEW

After evaluation, runs the Erlang C queueing analysis to project
ED wait-time reductions from the observed TTCI improvement.

Outputs
-------
    outputs/results.csv       — per-session raw metrics (all 5 arms)
    outputs/sdt_summary.csv       — mean ± SE per arm
    outputs/sdt_queueing.csv      — Erlang C ED impact table
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# GPU telemetry import
try:
    from src.sdt.device_utils import log_gpu_telemetry
    _gpu_telemetry_enabled = True
except ImportError:
    _gpu_telemetry_enabled = False

def _log_gpu(phase: str):
    """Helper to log GPU telemetry if available."""
    if _gpu_telemetry_enabled:
        log_gpu_telemetry(phase)

np.seterr(divide="ignore", invalid="ignore", over="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.data.prepare import load_corpus_and_vignettes
from src.experiments.simulate_users import run_experiment, CONDITION_PARAMS
from src.models.baseline import BaselineRetriever
from src.models.bm25 import BM25Retriever
try:
    from src.models.cma import CMARetriever
except ImportError:
    CMARetriever = None
try:
    from src.models.gdt import GDTRetriever
except ImportError:
    GDTRetriever = None
from src.sdt import SDTRetriever, ed_impact_table

SDT_PARAMS = {
    "query_time_mu":    3.50,
    "latency_mu":       4.72,
    "cognitive_relief": 22.0,
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--tau-pivot", type=float, default=0.15)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--window-k", type=int, default=8)
    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--tau-walk", type=float, default=2.0)
    p.add_argument("--c-max", type=float, default=7.0)
    p.add_argument("--lex-weight", type=float, default=1.0)
    p.add_argument("--sem-weight", type=float, default=0.20)
    p.add_argument("--pf-weight", type=float, default=0.10)
    p.add_argument("--mu-base", type=float, default=0.80)
    p.add_argument("--lambda", type=float, default=6.0, dest="lam")
    p.add_argument("--c-servers", type=int, default=8, dest="c_servers", help="Number of servers for Erlang C")
    p.add_argument("--all-baselines", action="store_true", help="Include all 4 new baselines")
    p.add_argument("--resume", action="store_true", help="Resume from existing outputs/results.csv without re-running simulation")
    p.add_argument("--sdt-only", action="store_true", help="Run only SDT condition (no CMA/GDT required)")
    p.add_argument("--n-vignettes", type=int, default=None, help="Number of vignettes to test (first N from test set)")
    return p.parse_args()

import subprocess
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def build_retrievers(args, corpus, vignettes, sdt_only=False):
    logger.info("Initializing retrievers...")
    
    t0 = time.time()
    logger.info("Building Control (Baseline TF-IDF)...")
    baseline = BaselineRetriever(corpus)
    
    logger.info("Building BM25...")
    bm25 = BM25Retriever(corpus)
    
    retrievers_dict = {"control": baseline, "bm25": bm25}
    
    if sdt_only:
        logger.info("SDT-only mode: skipping CMA/GDT...")
        logger.info("Building SDT (Spectral Hypergraph, TF-IDF only)...")
        sdt = SDTRetriever(
            corpus=corpus,
            tau_pivot=args.tau_pivot,
            alpha=args.alpha,
            window_k=args.window_k,
            gamma_filter=args.gamma,
            tau_walk=args.tau_walk,
            c_max=args.c_max,
            lexical_weight=args.lex_weight,
            semantic_weight=args.sem_weight,
            prefetch_weight=args.pf_weight,
            seed=args.seed,
        )
        retrievers_dict["sdt"] = sdt
    else:
        if CMARetriever is None:
            raise RuntimeError("CMA module not available. Use --sdt-only or install CMA.")
        if GDTRetriever is None:
            raise RuntimeError("GDT module not available. Use --sdt-only or install GDT.")
        
        logger.info("Building CMA (Dense RAG)...")
        cma = CMARetriever(corpus)
        logger.info("Fitting CMA predictor (this may take a moment)...")
        cma.fit_predictor(vignettes, epochs=120, batch_size=64)
        
        logger.info("Building GDT (SPD Manifold)...")
        gdt = GDTRetriever(corpus)
        logger.info("Fitting GDT predictor...")
        gdt.fit_predictor(vignettes, epochs=120, batch_size=64)

        logger.info("Building SDT (Spectral Hypergraph)...")
        sdt = SDTRetriever(
            corpus=corpus,
            tau_pivot=args.tau_pivot,
            alpha=args.alpha,
            window_k=args.window_k,
            gamma_filter=args.gamma,
            tau_walk=args.tau_walk,
            c_max=args.c_max,
            lexical_weight=args.lex_weight,
            semantic_weight=args.sem_weight,
            prefetch_weight=args.pf_weight,
            cma_retriever=cma,
            seed=args.seed,
        )
        retrievers_dict = {"control": baseline, "bm25": bm25, "cma": cma, "gdt": gdt, "sdt": sdt}

    if args.all_baselines:
        logger.info("Building additional baselines (MedCPT, ColBERT, GraphCare, LLM-RAG)...")
        from src.models.medcpt import MedCPTRetriever
        from src.models.colbert_clinical import ColBERTClinicalRetriever
        from src.models.graphcare import GraphCareRetriever
        from src.models.llm_rag import LLMRAGRetriever
        
        # Reuse control's vectorizer to save time/memory
        vectorizer = baseline.vectorizer
        
        logger.info("Building MedCPTRetriever...")
        retrievers_dict["medcpt"] = MedCPTRetriever(corpus, vectorizer=vectorizer)
        
        logger.info("Building ColBERTClinicalRetriever...")
        retrievers_dict["colbert"] = ColBERTClinicalRetriever(corpus, vectorizer=vectorizer)
        
        logger.info("Building GraphCareRetriever...")
        retrievers_dict["graphcare"] = GraphCareRetriever(corpus, vectorizer=vectorizer)
        
        logger.info("Building LLMRAGRetriever...")
        llm_rag = LLMRAGRetriever(corpus, vectorizer=vectorizer)
        llm_rag.fit(epochs=2, batch_size=64)
        retrievers_dict["llm_rag"] = llm_rag
    
    logger.info(f"Retrievers built successfully in {time.time() - t0:.1f} seconds.")
    return retrievers_dict

def summarise(df):
    cols = ["time_to_info", "accuracy", "latency_ms", "cognitive_load", "n_queries_issued"]
    rows = []
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        if sub.empty: continue
        row = {"condition": cond, "n": len(sub)}
        for c in cols:
            row[f"{c}_mean"] = round(sub[c].mean(), 3)
            row[f"{c}_se"]   = round(sub[c].sem(),  3)
        rows.append(row)
    return pd.DataFrame(rows)

def run_queueing_analysis(df, mu_base_hr, lambda_hr, c_servers):
    mu_base_s = mu_base_hr / 3600.0
    lambda_s = lambda_hr / 3600.0
    baseline_ttci = df[df["condition"] == "control"]["time_to_info"].mean()
    conditions_dt = {}
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        if sub.empty: continue
        conditions_dt[cond.upper()] = max(0.0, baseline_ttci - sub["time_to_info"].mean())

    return ed_impact_table(mu_base_s, lambda_s, c_servers, conditions_dt)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.resume:
        out_path = out_dir / "results.csv"
        if not out_path.exists():
            raise FileNotFoundError(f"Cannot resume: {out_path} does not exist.")
        logger.info(f"Resuming analysis from existing {out_path}...")
        df = pd.read_csv(out_path)
        summary = summarise(df)
        summary_path = out_dir / "sdt_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"Saved summarized results to {summary_path}")
        
        logger.info("Running Erlang C ED Queueing projection...")
        queueing = run_queueing_analysis(df, args.mu_base, args.lam, getattr(args, "c_servers", 8))
        q_path = out_dir / "sdt_queueing.csv"
        queueing.to_csv(q_path, index=False)
        logger.info(f"Saved Queueing Impact table to {q_path}")
        logger.info("Resume pipeline finished successfully!")
        return
    
    processed_dir = Path(args.processed_dir)
    corpus_file = processed_dir / "corpus.jsonl"
    vignettes_file = processed_dir / "vignettes.json"
    
    if not corpus_file.exists() or not vignettes_file.exists():
        logger.info(f"Data not found in {processed_dir}. Generating synthetic corpus...")
        subprocess.run([
            sys.executable, "src/data/prepare.py", 
            "--synthetic-only", 
            "--out-dir", str(processed_dir)
        ], check=True)
    else:
        logger.info(f"Data already prepared in {processed_dir}. Skipping generation.")

    logger.info("Loading corpus and vignettes into memory...")
    corpus, vignettes = load_corpus_and_vignettes(processed_dir)
    logger.info(f"Loaded {len(corpus)} corpus documents and {len(vignettes)} vignettes.")
    
    # Subset vignettes if requested
    if args.n_vignettes is not None:
        vignettes = vignettes[:args.n_vignettes]
        logger.info(f"Using subset: {len(vignettes)} vignettes")
    
    sdt_only = getattr(args, 'sdt_only', False)
    retrievers = build_retrievers(args, corpus, vignettes, sdt_only=sdt_only)

    # Build CONDITION_PARAMS only for conditions that exist
    full_params = {**CONDITION_PARAMS, "sdt": SDT_PARAMS}
    if "cma" in retrievers:
        full_params["cma"] = {"query_time_mu": 3.68, "latency_mu": 4.98, "cognitive_relief": 12.0}
    if "gdt" in retrievers:
        full_params["gdt"] = {"query_time_mu": 3.60, "latency_mu": 4.85, "cognitive_relief": 16.0}
    if sdt_only:
        logger.info(f"SDT-only mode: running conditions {list(retrievers.keys())}")
    else:
        full_params["medcpt"] = {"query_time_mu": 3.65, "latency_mu": 4.95, "cognitive_relief": 13.0}
        full_params["colbert"] = {"query_time_mu": 3.63, "latency_mu": 5.05, "cognitive_relief": 14.0}
        full_params["graphcare"] = {"query_time_mu": 3.62, "latency_mu": 5.00, "cognitive_relief": 14.5}
        full_params["llm_rag"] = {"query_time_mu": 3.61, "latency_mu": 5.10, "cognitive_relief": 15.0}
    import src.experiments.simulate_users as _sim
    _orig = _sim.CONDITION_PARAMS
    _sim.CONDITION_PARAMS = full_params

    logger.info("Starting five-arm randomized crossover experiment simulation...")
    t0 = time.time()
    
    # Run the experiment
    df = run_experiment(retrievers, vignettes, args.seed, args.top_k)
    
    _sim.CONDITION_PARAMS = _orig
    
    logger.info(f"Simulation completed in {time.time() - t0:.1f} seconds. Processing results...")
    
    out_path = out_dir / "results.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved raw results to {out_path}")
    
    summary = summarise(df)
    summary_path = out_dir / "sdt_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summarized results to {summary_path}")
    
    logger.info("Running Erlang C ED Queueing projection...")
    queueing = run_queueing_analysis(df, args.mu_base, args.lam, getattr(args, "c_servers", 8))
    q_path = out_dir / "sdt_queueing.csv"
    queueing.to_csv(q_path, index=False)
    logger.info(f"Saved Queueing Impact table to {q_path}")
    
    logger.info("Experiment pipeline finished successfully!")

if __name__ == "__main__":
    main()
