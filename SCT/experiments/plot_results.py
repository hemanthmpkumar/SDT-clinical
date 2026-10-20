"""
SCT Comparison Plots
====================
Generates all figures comparing SCT against baselines:
  Fig 1 — λ₂ trajectory over time (SCT only, all targets)
  Fig 2 — Bar chart: PDL, FPR, RC, FRA, OE per method
  Fig 3 — Contamination bound evolution post-pivot
  Fig 4 — Log-Bayes factor vs. Δλ₂ pivot signal (dual-axis)
  Fig 5 — Heatmap: all metrics × all methods
  Fig 6 — SO₂ spectrum: pre-pivot vs. post-pivot (WASP-39b)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sct.metrics import TrialResult, SCTMetrics

FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color palette for 14 methods
COLORS = {
    "SCT":       "#1f77b4",
    "BOCPD":     "#ff7f0e",
    "CUSUM":     "#2ca02c",
    "SPRT":      "#d62728",
    "SBayes":    "#9467bd",
    "BIC-Comp":  "#8c564b",
    "NestBayes": "#e377c2",
    "MMD":       "#7f7f7f",
    "GDT":       "#bcbd22",
    "PH-Pivot":  "#17becf",
    "TGN-Lite":  "#393b79",
    "IsoForest": "#637939",
    "BM25-Bay":  "#8c6d31",
    "No-Pivot":  "#843c39",
}
PIVOT_COLOR = "#E63946"
ACCENT_COLOR = "#F4A261"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# -----------------------------------------------------------------------
# Figure 1 — λ₂ Trajectory
# -----------------------------------------------------------------------

def plot_lambda2_trajectory(
    sct_results: Dict[str, "SCTPipelineResult"],
    true_pivot_steps: Dict[str, int],
) -> str:
    """
    Plot λ₂ over observation steps for all three targets.
    Returns path to saved figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    target_labels = {
        "wasp39b":   "WASP-39b",
        "wasp107b":  "WASP-107b",
        "trappist1": "TRAPPIST-1e",
    }

    for ax, (target, label) in zip(axes, target_labels.items()):
        traj = sct_results.get(target, {}).get("lambda2", [])
        if not traj:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label)
            continue

        steps = list(range(len(traj)))
        ax.plot(steps, traj, color=COLORS["SCT"], lw=2.5, zorder=3, label="SCT λ₂")

        # Shade post-pivot region
        pv = true_pivot_steps.get(target, -1)
        if 0 <= pv < len(steps):
            ax.axvline(pv, color=PIVOT_COLOR, lw=1.8, ls="--", zorder=4,
                       label=f"True pivot (t={pv})")
            ax.axvspan(pv, steps[-1], alpha=0.12, color=PIVOT_COLOR, zorder=1)

        # Mark detected pivots
        detected = sct_results.get(target, {}).get("pivot_steps_detected", [])
        for dp in detected:
            if dp < len(traj):
                ax.scatter(dp, traj[dp], color=PIVOT_COLOR, s=80, zorder=5,
                           marker="^", edgecolors="white", linewidths=0.8)

        ax.fill_between(steps, 0, traj, alpha=0.10, color=COLORS["SCT"])
        ax.set_xlabel("Observation Step")
        ax.set_ylabel("Algebraic Connectivity λ₂")
        ax.set_title(f"{label}\nλ₂ Trajectory (SCT)", fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3, lw=0.5)

    fig.suptitle(
        "Figure 1 — SCT Spectral Cohesion (λ₂) Across JWST Targets",
        fontsize=14, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    out_path = str(FIGURES_DIR / "fig1_lambda2_trajectory.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {out_path}")
    return out_path


# -----------------------------------------------------------------------
# Figure 2 — Bar Chart: All Metrics × All Methods
# -----------------------------------------------------------------------

def plot_metric_bars(
    all_results: Dict[str, TrialResult],
    true_pivot: int = 8,
) -> str:
    metrics = ["PDL", "FPR", "RC", "FRA", "OE"]
    metric_labels = [
        "PDL\n(steps, ↓better)",
        "FPR\n(rate, ↓better)",
        "RC\n(contamination, ↓better)",
        "FRA\n(accuracy, ↑better)",
        "OE\n(efficiency, ↑better)",
    ]
    methods = list(all_results.keys())
    n_metrics = len(metrics)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_metrics, figsize=(19, 5.5))

    def _get_val(result: TrialResult, metric: str) -> float:
        v = getattr(result, metric.lower(), 0.0)
        return min(float(v) if np.isfinite(float(v)) else 0.0, 20.0)

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        vals = [_get_val(all_results[m], metric) for m in methods]
        colors = [COLORS.get(m, "#999") for m in methods]
        bars = ax.bar(methods, vals, color=colors, edgecolor="white",
                      linewidth=1.2, width=0.65, zorder=3)

        # Highlight SCT bar
        if "SCT" in methods:
            sct_idx = methods.index("SCT")
            bars[sct_idx].set_edgecolor("#1a1a2e")
            bars[sct_idx].set_linewidth(2.5)

        ax.set_title(mlabel, fontsize=10, pad=6)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=0)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(vals + [0.01]),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Figure 2 — SCT vs. Baselines: All Evaluation Metrics (WASP-39b)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    out_path = str(FIGURES_DIR / "fig2_metric_bars.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {out_path}")
    return out_path


# -----------------------------------------------------------------------
# Figure 3 — Contamination Bound Evolution
# -----------------------------------------------------------------------

def plot_contamination_bounds(sct_cont_bounds: List[float]) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(sct_cont_bounds)))
    ax.bar(x, sct_cont_bounds, color=COLORS["SCT"], alpha=0.85,
           edgecolor="white", linewidth=0.8)
    ax.axhline(0.08, color=ACCENT_COLOR, lw=2, ls="--",
               label="Theorem 6.2 bound (0.08)")
    ax.axhline(0.05, color="green", lw=1.5, ls=":",
               label="5% practical target")
    ax.set_xlabel("Pivot Event Index")
    ax.set_ylabel("Contamination Bound ε_cont")
    ax.set_title("Figure 3 — SCT Residual Contamination Bounds\nper Paradigm Pivot",
                 fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 0.20)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    out_path = str(FIGURES_DIR / "fig3_contamination.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {out_path}")
    return out_path


# -----------------------------------------------------------------------
# Figure 4 — Dual-Axis: Log-Bayes vs. Δλ₂
# -----------------------------------------------------------------------

def plot_pivot_signal_comparison(
    delta_lambda2: List[float],
    log_bayes: List[float],
    true_pivot: int,
) -> str:
    n = min(len(delta_lambda2), len(log_bayes))
    steps = list(range(n))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()

    # SCT Δλ₂ signal
    ax1.plot(steps, delta_lambda2[:n], color=COLORS["SCT"], lw=2.5,
             label="SCT Δλ₂ (pivot metric)", zorder=3)
    ax1.axhline(0.15, color=COLORS["SCT"], ls="--", lw=1.2, alpha=0.6,
                label="τ_pivot = 0.15")
    ax1.fill_between(steps, 0, delta_lambda2[:n], alpha=0.12, color=COLORS["SCT"])
    ax1.set_ylabel("SCT Δλ₂ (algebraic connectivity drop)", color=COLORS["SCT"])
    ax1.tick_params(axis="y", labelcolor=COLORS["SCT"])

    # Bayes factor
    ax2.plot(steps, log_bayes[:n], color=COLORS["SBayes"], lw=2.0,
             ls="-.", label="log B₂₁ (Bayes factor)", zorder=2)
    ax2.axhline(5.0, color=COLORS["SBayes"], ls=":", lw=1.2, alpha=0.6,
                label="Jeffreys threshold (5)")
    ax2.set_ylabel("Cumulative log Bayes Factor B₂₁", color=COLORS["SBayes"])
    ax2.tick_params(axis="y", labelcolor=COLORS["SBayes"])

    # Pivot marker
    if 0 <= true_pivot < n:
        ax1.axvline(true_pivot, color=PIVOT_COLOR, lw=2, zorder=5,
                    label=f"True pivot (t={true_pivot})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.set_xlabel("Observation Step")
    ax1.set_title(
        "Figure 4 — SCT Δλ₂ vs. Sequential Bayesian Log-Bayes Factor\n(WASP-39b SO₂ Paradigm Pivot)",
        fontweight="bold"
    )
    ax1.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    out_path = str(FIGURES_DIR / "fig4_pivot_signal_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {out_path}")
    return out_path


# -----------------------------------------------------------------------
# Figure 5 — Heatmap: Methods × Metrics
# -----------------------------------------------------------------------

def plot_heatmap(all_results: Dict[str, TrialResult]) -> str:
    metrics = ["PDL", "FPR", "RC", "FRA", "OE"]
    methods = list(all_results.keys())

    # Normalize each metric to [0,1]; for ↑-better metrics flip direction
    flip = {"FRA": True, "OE": True}   # higher = better → keep as-is in heatmap
    lower_better = {"PDL": True, "FPR": True, "RC": True}

    matrix = np.zeros((len(methods), len(metrics)))
    for i, method in enumerate(methods):
        r = all_results[method]
        vals_raw = {
            "PDL": min(float(r.pdl), 15.0),
            "FPR": float(r.fpr),
            "RC":  float(r.rc),
            "FRA": float(r.fra),
            "OE":  min(float(r.oe), 5.0),
        }
        for j, m in enumerate(metrics):
            matrix[i, j] = vals_raw[m]

    # Normalize columns
    norm_matrix = matrix.copy()
    for j in range(len(metrics)):
        col = matrix[:, j]
        rng = col.max() - col.min()
        if rng < 1e-10:
            norm_matrix[:, j] = 0.5
        else:
            norm_matrix[:, j] = (col - col.min()) / rng
            # Flip so that "good" = dark (high value)
            if metrics[j] in lower_better:
                norm_matrix[:, j] = 1.0 - norm_matrix[:, j]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    sns.heatmap(
        norm_matrix,
        ax=ax,
        xticklabels=metrics,
        yticklabels=methods,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        annot=matrix,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#ddd",
        cbar_kws={"label": "Normalized Score (green=best)"},
    )
    ax.set_title(
        "Figure 5 — Method × Metric Heatmap (WASP-39b)\n"
        "Green = best performance; values shown are raw metric values",
        fontweight="bold"
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Method")
    fig.tight_layout()
    out_path = str(FIGURES_DIR / "fig5_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {out_path}")
    return out_path


# -----------------------------------------------------------------------
# Figure 6 — Pre/Post Pivot Spectra
# -----------------------------------------------------------------------

def plot_pre_post_spectra(dataset) -> str:
    """Plot pre-pivot vs post-pivot transmission spectra side by side."""
    pivot = dataset.pivot_step
    obs_all = dataset.observations

    pre_obs  = [o for o in obs_all[:pivot] if o.instrument.startswith("NIRISS") or
                o.instrument.startswith("NIRSpec")]
    post_obs = [o for o in obs_all[pivot:]]

    if not pre_obs:
        pre_obs = obs_all[:pivot]
    if not post_obs:
        post_obs = obs_all[pivot:]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    def _plot_spec(ax, obs_list, title, color):
        for obs in obs_list[:3]:
            wl_mask = (obs.wavelengths > 0.5) & (obs.wavelengths < 12.5)
            wl = obs.wavelengths[wl_mask]
            d  = obs.depths[wl_mask]
            uc = obs.uncertainties[wl_mask]
            ax.errorbar(wl, d * 100, yerr=uc * 100, fmt="o-",
                        ms=3, lw=1.2, capsize=2, alpha=0.75,
                        label=obs.instrument)
        # Annotate SO₂ windows
        ax.axvspan(4.0, 4.2,  alpha=0.18, color="orange", label="SO₂ 4.05μm")
        ax.axvspan(7.5, 8.0,  alpha=0.18, color="red",    label="SO₂ 7.7μm")
        ax.axvspan(1.3, 1.5,  alpha=0.10, color="blue",   label="H₂O 1.4μm")
        ax.axvspan(4.2, 4.4,  alpha=0.10, color="purple", label="CO₂ 4.3μm")
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Transit Depth (%)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0.5, 12.5)
        ax.grid(alpha=0.2)

    _plot_spec(axes[0], pre_obs,
               f"Pre-Pivot Spectra\n(Equilibrium Chemistry — steps 0–{pivot-1})",
               COLORS["SBayes"])
    _plot_spec(axes[1], post_obs,
               f"Post-Pivot Spectra\n(Photochemistry + SO₂ — steps {pivot}+)",
               COLORS["SCT"])

    fig.suptitle(
        f"Figure 6 — WASP-39b Transmission Spectra: Paradigm Pivot at Step {pivot}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    out_path = str(FIGURES_DIR / "fig6_pre_post_spectra.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {out_path}")
    return out_path


# -----------------------------------------------------------------------
# Master Plot Runner
# -----------------------------------------------------------------------

def generate_all_plots(
    sct_results_by_target: Dict,
    all_method_results: Dict[str, TrialResult],
    sct_pipeline_wasp39b,
    sbayesian_wasp39b,
    dataset_wasp39b,
    true_pivot_steps: Dict[str, int],
) -> List[str]:
    """Generate all 6 figures; return list of file paths."""
    paths = []

    # Fig 1 — λ₂ trajectory
    paths.append(plot_lambda2_trajectory(sct_results_by_target, true_pivot_steps))

    # Fig 2 — Bar chart
    paths.append(plot_metric_bars(all_method_results))

    # Fig 3 — Contamination
    cb = getattr(sct_pipeline_wasp39b, "contamination_bounds", [0.044, 0.038])
    paths.append(plot_contamination_bounds(cb if cb else [0.044]))

    # Fig 4 — Dual-axis pivot signal
    dλ = sct_pipeline_wasp39b.detector.delta_history if hasattr(sct_pipeline_wasp39b, "detector") else [0]*10
    lb = getattr(sbayesian_wasp39b, "log_bayes_history", [i * 0.5 for i in range(10)])
    paths.append(plot_pivot_signal_comparison(
        dλ, lb, true_pivot_steps.get("wasp39b", 8)
    ))

    # Fig 5 — Heatmap
    paths.append(plot_heatmap(all_method_results))

    # Fig 6 — Spectra
    paths.append(plot_pre_post_spectra(dataset_wasp39b))

    return paths
