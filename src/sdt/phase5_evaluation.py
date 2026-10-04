#!/usr/bin/env python3
"""
src/sdt/phase5_evaluation.py

Phase 5 — Evaluation & Operational Analytics
=============================================

Two distinct responsibilities:

A. Retrieval Metrics
   ─────────────────
   • TTCI  — Time-to-Correct-Information (seconds, lower is better)
   • MRR   — Mean Reciprocal Rank (higher is better)
   • CP    — Contextual Precision = |V_active ∩ V*_relevant| / |V_active|
   • Pivot-F1 — Precision/Recall of interference gate fires vs. ground truth

B. Erlang C Queueing Analysis
   ──────────────────────────
   Maps retrieval latency reductions Δt to non-linear ED wait time reductions
   via the M/M/c (Erlang C) formula (paper §7.4, Eq. 25–29).

   Service rate acceleration:
       μ' = 1 / (μ⁻¹ − Δt)

   Traffic intensity:
       ρ' = λ / (c · μ')

   Erlang C waiting probability:
       P_W  = [(cρ')^c / c! · 1/(1−ρ')] /
              [Σ_{k=0}^{c-1}(cρ')^k/k! + (cρ')^c/c!·1/(1−ρ')]

   Mean queue wait:
       W_q = P_W · (μ')⁻¹ / (c · (1−ρ'))

References
----------
Erlang (1917)   "Solution of Some Problems in the Theory of Probabilities"
Kleinrock (1975) "Queueing Systems, Vol 1"
Gross & Harris (2008) "Fundamentals of Queueing Theory"
Armony et al. (2015) "On Patient Flow in Hospitals"
Green (2006)    "Queueing Analysis in Healthcare"
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===========================================================================
# A. Retrieval Metrics
# ===========================================================================

@dataclass
class PivotEvent:
    """Ground-truth or predicted diagnostic pivot event."""
    turn: int
    vignette_id: str
    is_ground_truth: bool = True


@dataclass
class QueryResult:
    """Result of a single retrieval query."""
    turn: int
    query_text: str
    retrieved_ids: list[str]
    target_id: str
    timestamp_s: float               # wall-clock time at which result arrived
    active_node_ids: list[str]       # V_active post-Fiedler cut at this turn
    relevant_node_ids: list[str]     # V*_relevant ground truth for this turn
    gate_fired: bool = False


class SDTEvaluator:
    """Compute all evaluation metrics for SDT (and baselines).

    Usage
    -----
    evaluator = SDTEvaluator()
    evaluator.add_result(query_result)
    ...
    metrics = evaluator.compute()
    """

    def __init__(self):
        self._results: list[QueryResult] = []
        self._gt_pivots: list[PivotEvent] = []
        self._pred_pivots: list[PivotEvent] = []

    def add_result(self, result: QueryResult) -> None:
        self._results.append(result)

    def add_ground_truth_pivot(self, turn: int, vignette_id: str) -> None:
        self._gt_pivots.append(PivotEvent(turn, vignette_id, True))

    def add_predicted_pivot(self, turn: int, vignette_id: str) -> None:
        self._pred_pivots.append(PivotEvent(turn, vignette_id, False))

    # ------------------------------------------------------------------
    # TTCI — Time-to-Correct-Information
    # ------------------------------------------------------------------

    def ttci(self) -> float:
        """Mean time (s) until the target note was in the top-k results.

        For each query, the TTCI is the timestamp at which the correct
        document first appears.  If the target was never retrieved,
        a penalty of 300 s (right-censor limit) is applied.
        """
        times = []
        for r in self._results:
            if r.target_id in r.retrieved_ids:
                times.append(r.timestamp_s)
            else:
                times.append(300.0)      # right-censor penalty
        return float(np.mean(times)) if times else 0.0

    # ------------------------------------------------------------------
    # MRR — Mean Reciprocal Rank
    # ------------------------------------------------------------------

    def mrr(self) -> float:
        """Mean Reciprocal Rank of the target note across all queries."""
        rrs = []
        for r in self._results:
            try:
                rank = r.retrieved_ids.index(r.target_id) + 1
                rrs.append(1.0 / rank)
            except ValueError:
                rrs.append(0.0)
        return float(np.mean(rrs)) if rrs else 0.0

    # ------------------------------------------------------------------
    # CP — Contextual Precision
    # ------------------------------------------------------------------

    def contextual_precision(self) -> float:
        """CP = |V_active ∩ V*_relevant| / |V_active| averaged over turns."""
        cps = []
        for r in self._results:
            v_active = set(r.active_node_ids)
            v_relevant = set(r.relevant_node_ids)
            if len(v_active) == 0:
                continue
            cp = len(v_active & v_relevant) / len(v_active)
            cps.append(cp)
        return float(np.mean(cps)) if cps else 0.0

    # ------------------------------------------------------------------
    # Pivot Detection F1
    # ------------------------------------------------------------------

    def pivot_f1(self, tolerance_turns: int = 1) -> dict[str, float]:
        """Precision, Recall, F1 of gate fires vs. ground-truth pivots.

        A predicted pivot is a true positive if it falls within
        ±tolerance_turns of a ground-truth pivot.

        Returns
        -------
        dict with keys: precision, recall, f1
        """
        gt_turns  = set(e.turn for e in self._gt_pivots)
        pred_turns = set(e.turn for e in self._pred_pivots)

        tp = sum(
            any(abs(pt - gt) <= tolerance_turns for gt in gt_turns)
            for pt in pred_turns
        )
        precision = tp / max(len(pred_turns), 1)
        recall    = tp / max(len(gt_turns), 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------------
    # Composite metric report
    # ------------------------------------------------------------------

    def compute(self) -> dict[str, float]:
        """Return all metrics as a flat dictionary."""
        pivot_metrics = self.pivot_f1()
        return {
            "ttci":                 self.ttci(),
            "mrr":                  self.mrr(),
            "contextual_precision": self.contextual_precision(),
            "pivot_precision":      pivot_metrics["precision"],
            "pivot_recall":         pivot_metrics["recall"],
            "pivot_f1":             pivot_metrics["f1"],
            "n_queries":            len(self._results),
            "n_gt_pivots":          len(self._gt_pivots),
            "n_pred_pivots":        len(self._pred_pivots),
        }

    def reset(self) -> None:
        self._results.clear()
        self._gt_pivots.clear()
        self._pred_pivots.clear()


# ===========================================================================
# B. Erlang C Queueing Model
# ===========================================================================

def erlang_c(c: int, rho: float) -> float:
    """Compute the Erlang C waiting probability P_W.

    P_W is the probability that an arriving patient has to wait
    (all c servers busy) in an M/M/c queue.

    Args:
        c   : number of servers (clinicians / beds).
        rho : traffic intensity ρ = λ / (c·μ). Must be < 1.

    Returns:
        P_W : float in [0, 1).
    """
    if rho >= 1.0:
        return 1.0
    a = c * rho        # offered load (Erlangs)

    # Numerator: a^c / c! · 1/(1−ρ)
    numerator = (a ** c) / math.factorial(c) * (1.0 / (1.0 - rho))

    # Denominator: Σ_{k=0}^{c-1} a^k/k!  +  numerator
    summation = sum((a ** k) / math.factorial(k) for k in range(c))
    denominator = summation + numerator

    return float(numerator / denominator) if denominator > 1e-15 else 0.0


def compute_wq(
    mu_base: float,
    lambda_arrival: float,
    c_servers: int,
    delta_t_s: float = 0.0,
) -> dict[str, float]:
    """Compute mean queue wait W_q and related ED metrics.

    Args:
        mu_base        : baseline service rate μ (patients / second).
        lambda_arrival : patient arrival rate λ (patients / second).
        c_servers      : number of active clinicians/beds.
        delta_t_s      : retrieval latency saving Δt (seconds per patient).
                         Set 0 for baseline (no AI).

    Returns
    -------
    dict with keys:
        mu_prime   : accelerated service rate μ' (patients/s).
        rho_prime  : updated traffic intensity ρ'.
        P_W        : Erlang C waiting probability.
        W_q_s      : mean queue wait in seconds.
        W_q_min    : mean queue wait in minutes.
    """
    # Accelerated service rate (Eq. 25):  μ' = 1 / (μ⁻¹ − Δt)
    service_time_base = 1.0 / mu_base   # seconds per patient
    service_time_new  = max(service_time_base - delta_t_s, 1e-6)
    mu_prime = 1.0 / service_time_new

    # Traffic intensity (Eq. 26): ρ' = λ / (c · μ')
    rho_prime = lambda_arrival / (c_servers * mu_prime)
    rho_prime = min(rho_prime, 0.9999)   # cap to keep stable

    # Erlang C (Eq. 27)
    P_W = erlang_c(c_servers, rho_prime)

    # Mean queue wait (Eq. 28): W_q = P_W / (c · μ' · (1 − ρ'))
    denom = c_servers * mu_prime * (1.0 - rho_prime)
    W_q_s = float(P_W / denom) if denom > 1e-12 else 9999.0

    return {
        "mu_base":   mu_base,
        "delta_t_s": delta_t_s,
        "mu_prime":  mu_prime,
        "rho_prime": rho_prime,
        "P_W":       P_W,
        "W_q_s":     W_q_s,
        "W_q_min":   W_q_s / 60.0,
    }


# ---------------------------------------------------------------------------
# Convenience: build the ED impact table (Table 3 from paper)
# ---------------------------------------------------------------------------

def ed_impact_table(
    mu_base: float,
    lambda_arrival: float,
    c_servers: int,
    conditions: dict[str, float],
) -> pd.DataFrame:
    """Generate the ED queueing impact table for multiple systems.

    Args:
        mu_base        : baseline service rate (patients/s).
        lambda_arrival : patient arrival rate (patients/s).
        c_servers      : number of ED servers.
        conditions     : dict mapping condition name → Δt (seconds saved).
                         E.g. {"Baseline": 0.0, "GDT": 3.2, "SDT": 8.3}

    Returns:
        DataFrame with columns: system, mu_prime_hr, rho_prime, W_q_min.
    """
    rows = []
    for system_name, delta_t in conditions.items():
        result = compute_wq(mu_base, lambda_arrival, c_servers, delta_t)
        rows.append({
            "System":          system_name,
            "Δt (s)":          round(delta_t, 1),
            "μ' (hr⁻¹)":      round(result["mu_prime"] * 3600, 2),
            "ρ'":              round(result["rho_prime"], 3),
            "P_W":             round(result["P_W"], 3),
            "W_q (min)":       round(result["W_q_min"], 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Erlang C ED Impact (c=8, λ=6 hr⁻¹) ===\n")
    mu_base        = 0.80 / 3600      # 0.80 patients/hr → /s
    lambda_arrival = 6.0 / 3600      # 6 arrivals/hr → /s
    c_servers      = 8

    conditions = {
        "Baseline (no AI)": 0.0,
        "GDT":              3.2,
        "SDT":              8.3,
    }
    df = ed_impact_table(mu_base, lambda_arrival, c_servers, conditions)
    print(df.to_string(index=False))
    print()

    print("=== Sample Retrieval Metrics ===")
    ev = SDTEvaluator()
    ev.add_result(QueryResult(
        turn=1, query_text="sepsis workup",
        retrieved_ids=["note_001", "note_042", "note_017"],
        target_id="note_042",
        timestamp_s=5.1,
        active_node_ids=["n1", "n2", "n3", "n4"],
        relevant_node_ids=["n2", "n4"],
        gate_fired=False,
    ))
    ev.add_result(QueryResult(
        turn=2, query_text="renal dosing creatinine",
        retrieved_ids=["note_017", "note_099"],
        target_id="note_099",
        timestamp_s=4.8,
        active_node_ids=["n3", "n5", "n6"],
        relevant_node_ids=["n5", "n6"],
        gate_fired=True,
    ))
    ev.add_ground_truth_pivot(turn=2, vignette_id="v01")
    ev.add_predicted_pivot(turn=2, vignette_id="v01")
    metrics = ev.compute()
    for k, v in metrics.items():
        print(f"  {k:28s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
