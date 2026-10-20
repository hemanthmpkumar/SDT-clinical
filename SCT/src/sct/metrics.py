"""
Phase 5 — Evaluation Metrics (SCT)
====================================
Implements all five metrics from the proposal:
  PDL  – Pivot Detection Latency
  FPR  – False Positive Rate
  RC   – Residual Contamination
  FRA  – Final Retrieval Accuracy
  OE   – Observation Efficiency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrialResult:
    """Metrics for a single experimental trial."""
    method: str
    pdl: float                    # Pivot detection latency (steps)
    fpr: float                    # False positive rate
    rc: float                     # Residual contamination
    fra: float                    # Final retrieval accuracy ∈ [0,1]
    oe: float                     # Observation efficiency (ratio)
    n_obs_to_90pct: int           # Obs. needed for 90% FRA
    lambda2_trajectory: List[float] = field(default_factory=list)
    pivot_steps_detected: List[int] = field(default_factory=list)
    contamination_bounds: List[float] = field(default_factory=list)


class SCTMetrics:
    """
    Computes and aggregates SCT evaluation metrics across multiple trials.
    """

    @staticmethod
    def pivot_detection_latency(
        detected_steps: List[int],
        true_pivot_step: int,
    ) -> float:
        """PDL = first detected step - true pivot step. Negative = early."""
        if not detected_steps:
            return float("inf")
        # Closest detected pivot after true pivot
        after = [s for s in detected_steps if s >= true_pivot_step]
        if after:
            return float(min(after) - true_pivot_step)
        # If only false positives before true pivot
        return float(max(detected_steps) - true_pivot_step)

    @staticmethod
    def false_positive_rate(
        detected_steps: List[int],
        true_pivot_step: int,
        total_non_pivot_steps: int,
    ) -> float:
        """FPR = |false positive detections| / |non-pivot steps|."""
        if total_non_pivot_steps == 0:
            return 0.0
        false_pos = sum(1 for s in detected_steps if abs(s - true_pivot_step) > 2)
        return false_pos / total_non_pivot_steps

    @staticmethod
    def residual_contamination(
        active_embeddings: np.ndarray,
        stale_embeddings: np.ndarray,
    ) -> float:
        """
        RC = cosine similarity between mean active and mean stale embeddings.
        Lower is better (cleaner separation).
        """
        if active_embeddings.shape[0] == 0 or stale_embeddings.shape[0] == 0:
            return 0.0
        e_active = active_embeddings.mean(axis=0)
        e_stale  = stale_embeddings.mean(axis=0)
        denom = np.linalg.norm(e_active) * np.linalg.norm(e_stale)
        if denom < 1e-10:
            return 0.0
        return float(abs(np.dot(e_active, e_stale) / denom))

    @staticmethod
    def final_retrieval_accuracy(
        estimated_params: np.ndarray,
        reference_params: np.ndarray,
    ) -> float:
        """
        FRA = 1 - ||θ_est - θ_ref|| / ||θ_ref||
        Clamped to [0, 1].
        """
        denom = np.linalg.norm(reference_params)
        if denom < 1e-10:
            return 1.0 if np.linalg.norm(estimated_params) < 1e-10 else 0.0
        fra = 1.0 - np.linalg.norm(estimated_params - reference_params) / denom
        return float(np.clip(fra, 0.0, 1.0))

    @staticmethod
    def observation_efficiency(
        n_obs_baseline: int,
        n_obs_sct: int,
    ) -> float:
        """OE = N_baseline / N_sct. >1 means SCT is more efficient."""
        if n_obs_sct <= 0:
            return float("inf")
        return n_obs_baseline / n_obs_sct

    @staticmethod
    def aggregate(results: List[TrialResult]) -> Dict:
        """Compute mean ± std for all metrics across trials."""
        if not results:
            return {}

        def _stats(vals):
            arr = np.array(vals, dtype=float)
            finite = arr[np.isfinite(arr)]
            return {
                "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                "std":  float(np.std(finite))  if len(finite) else float("nan"),
                "median": float(np.median(finite)) if len(finite) else float("nan"),
            }

        return {
            "PDL":  _stats([r.pdl  for r in results]),
            "FPR":  _stats([r.fpr  for r in results]),
            "RC":   _stats([r.rc   for r in results]),
            "FRA":  _stats([r.fra  for r in results]),
            "OE":   _stats([r.oe   for r in results]),
        }

    @staticmethod
    def summary_table(
        results_by_method: Dict[str, List[TrialResult]],
    ) -> str:
        """Return a formatted ASCII comparison table."""
        header = f"{'Method':<22} {'PDL':>8} {'FPR':>8} {'RC':>8} {'FRA':>8} {'OE':>8}"
        lines = [header, "-" * len(header)]
        for method, results in results_by_method.items():
            agg = SCTMetrics.aggregate(results)
            if not agg:
                continue
            row = (
                f"{method:<22} "
                f"{agg['PDL']['mean']:>7.2f} "
                f"{agg['FPR']['mean']:>7.3f} "
                f"{agg['RC']['mean']:>7.3f} "
                f"{agg['FRA']['mean']:>7.3f} "
                f"{agg['OE']['mean']:>7.2f}"
            )
            lines.append(row)
        return "\n".join(lines)
