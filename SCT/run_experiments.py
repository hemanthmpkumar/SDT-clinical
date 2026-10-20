#!/usr/bin/env python3
"""
SCT Main Experiment Runner
===========================
Orchestrates the complete 5-step pipeline:
  1. Download / generate datasets (WASP-39b, WASP-107b, TRAPPIST-1e)
  2. Train / fit SCT on each target
  3. Run all baseline comparisons
  4. Compute and print metrics comparison table
  5. Generate all 6 comparison figures

Usage:
    cd /Users/hemanth/Projects/Papers/SDT/SCT
    python run_experiments.py [--synthetic] [--verbose]

Flags:
    --synthetic   Force use of synthetic data even if cache exists
    --verbose     Print detailed step-by-step SCT progress
    --target      Run only one target: wasp39b | wasp107b | trappist1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---- Path setup -------------------------------------------------------
ROOT = Path(__file__).parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

# ---- SCT imports -------------------------------------------------------
from sct.pipeline import SCTPipeline
from sct.metrics import SCTMetrics, TrialResult
from experiments.download_data import download_all, SCTDataset
from experiments.baselines import run_all_baselines
from experiments.plot_results import generate_all_plots


OUTPUTS_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def print_banner(text: str):
    w = 70
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


def save_metrics_csv(results_by_method: dict, target: str):
    """Save all metric results to CSV."""
    path = METRICS_DIR / f"{target}_metrics.csv"
    rows = []
    for method, result in results_by_method.items():
        rows.append({
            "target": target,
            "method": method,
            "PDL":    f"{result.pdl:.4f}",
            "FPR":    f"{result.fpr:.4f}",
            "RC":     f"{result.rc:.4f}",
            "FRA":    f"{result.fra:.4f}",
            "OE":     f"{result.oe:.4f}",
            "n_obs":  result.n_obs_to_90pct,
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] Saved → {path}")
    return str(path)


def save_metrics_json(results_by_method: dict, target: str):
    """Save all metric results to JSON for downstream analysis."""
    path = METRICS_DIR / f"{target}_metrics.json"
    data = {}
    for method, result in results_by_method.items():
        data[method] = {
            "PDL": result.pdl,
            "FPR": result.fpr,
            "RC":  result.rc,
            "FRA": result.fra,
            "OE":  result.oe,
            "n_obs_to_90pct": result.n_obs_to_90pct,
            "pivot_steps_detected": result.pivot_steps_detected,
            "lambda2_trajectory": result.lambda2_trajectory,
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[JSON] Saved → {path}")
    return str(path)


# -----------------------------------------------------------------------
# Single-target experiment
# -----------------------------------------------------------------------

def run_target_experiment(
    target_name: str,
    dataset: SCTDataset,
    verbose: bool = True,
) -> dict:
    """
    Run SCT + all baselines on one dataset; return dict of TrialResults.
    """
    print_banner(f"Step 2–4: {dataset.target_name} ({len(dataset.observations)} observations)")
    print(f"  True pivot at step: {dataset.pivot_step}")

    # ---- Step 2: SCT ----
    print("\n[Step 2] Running SCT pipeline...")
    t0 = time.time()
    pipeline = SCTPipeline(
        tau_pivot=0.08,
        window_size=25,
        budget=10.0,
        adaptive_threshold=True,
        verbose=verbose,
    )
    sct_result = pipeline.run(dataset)
    sct_elapsed = time.time() - t0
    print(f"  SCT completed in {sct_elapsed:.2f}s")

    # ---- Step 3: Baselines ----
    print("\n[Step 3] Running baseline comparisons...")
    baseline_results = run_all_baselines(dataset)

    # ---- Step 4: Combine & print ----
    all_results = {"SCT": sct_result, **baseline_results}

    print("\n[Step 4] Metric Comparison Table:")
    print(SCTMetrics.summary_table({k: [v] for k, v in all_results.items()}))

    # Save artifacts
    save_metrics_csv(all_results, target_name)
    save_metrics_json(all_results, target_name)

    return {
        "pipeline": pipeline,
        "results": all_results,
        "sct_result": sct_result,
    }


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SCT Experiment Runner")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data generation")
    parser.add_argument("--verbose",   action="store_true",
                        help="Verbose SCT step-by-step output")
    parser.add_argument("--target",    type=str, default="all",
                        choices=["all","wasp39b","wasp107b","hd209458b",
                                 "wasp96b","55cnce","trappist1b"],
                        help="Which target(s) to run")
    parser.add_argument("--no-plots",  action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    start_time = time.time()

    # =========================================================
    # Step 1: Download / Generate Data
    # =========================================================
    print_banner("Step 1: Downloading / Generating Datasets")
    datasets = download_all(force_synthetic=args.synthetic)
    for name, ds in datasets.items():
        print(f"  ✓ {name:12s} — {len(ds.observations)} obs, pivot @ step {ds.pivot_step}")

    if args.target != "all":
        targets = {args.target: datasets[args.target]}
    else:
        targets = datasets

    # =========================================================
    # Steps 2–4: SCT + Baselines per Target
    # =========================================================
    target_experiments = {}
    for target_name, dataset in targets.items():
        experiment = run_target_experiment(
            target_name, dataset, verbose=args.verbose
        )
        target_experiments[target_name] = experiment

    # =========================================================
    # Step 5: Generate Comparison Plots
    # =========================================================
    if not args.no_plots:
        print_banner("Step 5: Generating Comparison Figures")

        # Collect SCT trajectories
        sct_results_by_target = {}
        for target_name, exp in target_experiments.items():
            pipe = exp["pipeline"]
            sct_results_by_target[target_name] = {
                "lambda2": pipe.lambda2_trajectory,
                "pivot_steps_detected": pipe.pivot_steps_detected,
            }

        # Pick primary target for multi-method comparison plots
        primary_target = next(
            (k for k in ["wasp39b","wasp107b","hd209458b","wasp96b","55cnce","trappist1b"]
             if k in target_experiments),
            list(target_experiments.keys())[0]
        )
        wasp39b_exp = target_experiments[primary_target]
        all_method_results = wasp39b_exp["results"]

        # Pipeline for primary target (Figs 3, 4, 6)
        wasp39b_pipeline = wasp39b_exp["pipeline"]
        sbayesian_wasp39b = None
        if "SBayes" in all_method_results:
            from experiments.baselines import SequentialBayes
            sb = SequentialBayes()
            sb.run(datasets.get(primary_target, list(datasets.values())[0]))
            sbayesian_wasp39b = sb

        true_pivot_steps = {
            name: ds.pivot_step for name, ds in datasets.items()
        }

        figure_paths = generate_all_plots(
            sct_results_by_target=sct_results_by_target,
            all_method_results=all_method_results,
            sct_pipeline_wasp39b=wasp39b_exp["pipeline"],
            sbayesian_wasp39b=sbayesian_wasp39b,
            dataset_wasp39b=datasets.get(primary_target, list(datasets.values())[0]),
            true_pivot_steps=true_pivot_steps,
        )

        print(f"\n  Generated {len(figure_paths)} figures:")
        for p in figure_paths:
            print(f"    → {p}")

    # =========================================================
    # Final Summary
    # =========================================================
    elapsed = time.time() - start_time
    print_banner(f"Pipeline Complete — {elapsed:.1f}s total")
    print("  Outputs:")
    print(f"    Figures : {OUTPUTS_DIR / 'figures'}/")
    print(f"    Metrics : {METRICS_DIR}/")

    # Print final SCT summary for primary target
    wexp = target_experiments.get(
        next((k for k in ["wasp39b","wasp107b","hd209458b","wasp96b","55cnce","trappist1b"]
              if k in target_experiments), list(target_experiments.keys())[0]))
    sct = wexp["results"]["SCT"]
    prim = next((k for k in ["wasp39b","wasp107b","hd209458b","wasp96b","55cnce","trappist1b"]
                 if k in target_experiments), list(target_experiments.keys())[0])
    print(f"\n  SCT results for primary target ({prim}):")
    print(f"    PDL:  SCT={sct.pdl:.1f}   (lower=better)")
    print(f"    RC:   SCT={sct.rc:.3f}  (lower=better)")
    print(f"    FRA:  SCT={sct.fra:.3f}  (higher=better)")
    print(f"    OE:   SCT={sct.oe:.2f}   (higher=better)")


if __name__ == "__main__":
    main()
