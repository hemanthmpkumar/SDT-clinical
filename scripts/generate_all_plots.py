#!/usr/bin/env python3
"""
scripts/generate_all_plots.py

Regenerate all publication-quality figures for latex/IJDS/figures/ and other targets
from the complete 9-arm crossover experiment results (outputs/results.csv, sdt_summary.csv, sdt_queueing.csv).
All figures follow a strict Grayscale-Safe Design (distinct monochrome/gray tones,
geometric hatching patterns, varied linestyles, and unique markers) ensuring 100%
legibility and contrast in black-and-white print while remaining crisp and elegant in digital format.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Publication plot styling - Grayscale-safe
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['hatch.color'] = '#1a1a1a'
plt.rcParams['hatch.linewidth'] = 0.9

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRS = [
    ROOT / "latex" / "IJDS" / "figures",
    ROOT / "latex" / "figures",
    ROOT / "outputs" / "plots",
    ROOT / "reduced" / "figures"
]

for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

df_results = pd.read_csv(ROOT / "outputs" / "results.csv")
df_summary = pd.read_csv(ROOT / "outputs" / "sdt_summary.csv")
df_queue = pd.read_csv(ROOT / "outputs" / "sdt_queueing.csv")

# Standard condition order and display labels
COND_ORDER = ["control", "bm25", "llm_rag", "colbert", "cma", "medcpt", "graphcare", "gdt", "sdt"]
COND_LABELS = {
    "control": "Control",
    "bm25": "BM25",
    "llm_rag": "LLM-RAG",
    "colbert": "ColBERT",
    "cma": "CMA",
    "medcpt": "MedCPT",
    "graphcare": "GraphCare",
    "gdt": "GDT",
    "sdt": "SDT (Ours)"
}

# Grayscale-safe styles: progression of grayscale tones + distinct geometric hatches
GRAY_STYLES = {
    "control":   {"color": "#f2f2f2", "hatch": ""},
    "bm25":      {"color": "#e0e0e0", "hatch": "///"},
    "llm_rag":   {"color": "#d0d0d0", "hatch": "\\\\\\"},
    "colbert":   {"color": "#bfbfbf", "hatch": "xxx"},
    "cma":       {"color": "#adadad", "hatch": "---"},
    "medcpt":    {"color": "#9b9b9b", "hatch": "+++"},
    "graphcare": {"color": "#888888", "hatch": "ooo"},
    "gdt":       {"color": "#585858", "hatch": "//////"},
    "sdt":       {"color": "#181818", "hatch": ""}   # Solid deep charcoal/black anchor
}

def save_fig(fig, filename):
    for d in OUTPUT_DIRS:
        fig.savefig(d / filename)

# -----------------------------------------------------------------------------
# 1. TTCI Bar Chart
# -----------------------------------------------------------------------------
def plot_ttci():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    
    sub = df_summary.set_index("condition").reindex(COND_ORDER).reset_index()
    labels = [COND_LABELS[c] for c in sub["condition"]]
    
    for i, row in sub.iterrows():
        c = row["condition"]
        st = GRAY_STYLES[c]
        ax.bar(
            i,
            row["time_to_info_mean"],
            yerr=row["time_to_info_se"],
            color=st["color"],
            hatch=st["hatch"],
            edgecolor="#000000",
            linewidth=1.2 if c == "sdt" else 0.85,
            capsize=3.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#000000"}
        )
    
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Time to Correct Information (seconds)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cccccc")
    
    # Highlight reduction vs control
    ctrl_ttci = sub.loc[sub["condition"] == "control", "time_to_info_mean"].values[0]
    ax.axhline(ctrl_ttci, color="#333333", linestyle=":", linewidth=1.2, alpha=0.8)
    
    plt.tight_layout()
    save_fig(fig, "fig_ttci_bar.png")
    plt.close()
    print("-> Generated fig_ttci_bar.png (grayscale-safe)")

# -----------------------------------------------------------------------------
# 2. Accuracy Bar Chart
# -----------------------------------------------------------------------------
def plot_accuracy():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    
    sub = df_summary.set_index("condition").reindex(COND_ORDER).reset_index()
    labels = [COND_LABELS[c] for c in sub["condition"]]
    
    for i, row in sub.iterrows():
        c = row["condition"]
        st = GRAY_STYLES[c]
        ax.bar(
            i,
            row["accuracy_mean"],
            yerr=row["accuracy_se"],
            color=st["color"],
            hatch=st["hatch"],
            edgecolor="#000000",
            linewidth=1.2 if c == "sdt" else 0.85,
            capsize=3.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#000000"}
        )
    
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Diagnostic Resolution Accuracy", fontsize=11)
    ax.set_ylim(0.70, 1.00)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cccccc")
    
    ctrl_acc = sub.loc[sub["condition"] == "control", "accuracy_mean"].values[0]
    ax.axhline(ctrl_acc, color="#333333", linestyle=":", linewidth=1.2, alpha=0.8)
    
    plt.tight_layout()
    save_fig(fig, "fig_accuracy_bar.png")
    plt.close()
    print("-> Generated fig_accuracy_bar.png (grayscale-safe)")

# -----------------------------------------------------------------------------
# 3. Cognitive Load Bar Chart
# -----------------------------------------------------------------------------
def plot_cognitive_load():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    
    sub = df_summary.set_index("condition").reindex(COND_ORDER).reset_index()
    labels = [COND_LABELS[c] for c in sub["condition"]]
    
    for i, row in sub.iterrows():
        c = row["condition"]
        st = GRAY_STYLES[c]
        ax.bar(
            i,
            row["cognitive_load_mean"],
            yerr=row["cognitive_load_se"],
            color=st["color"],
            hatch=st["hatch"],
            edgecolor="#000000",
            linewidth=1.2 if c == "sdt" else 0.85,
            capsize=3.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#000000"}
        )
    
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Cumulative Cognitive Load Score", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cccccc")
    
    ctrl_load = sub.loc[sub["condition"] == "control", "cognitive_load_mean"].values[0]
    ax.axhline(ctrl_load, color="#333333", linestyle=":", linewidth=1.2, alpha=0.8)
    
    plt.tight_layout()
    save_fig(fig, "fig_cognitive_load_bar.png")
    plt.close()
    print("-> Generated fig_cognitive_load_bar.png (grayscale-safe)")

# -----------------------------------------------------------------------------
# 4. Subgroup Accuracy by Complexity
# -----------------------------------------------------------------------------
def plot_subgroup_accuracy():
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    
    acc_comp = df_results.groupby(['condition', 'complexity'])['accuracy'].mean().unstack()
    acc_comp = acc_comp.reindex(COND_ORDER)
    
    x = np.arange(len(COND_ORDER))
    width = 0.36
    
    rects1 = ax.bar(x - width/2, acc_comp['low'], width, label='Low Complexity',
                    color='#e4e4e4', edgecolor='#000000', linewidth=0.85)
    rects2 = ax.bar(x + width/2, acc_comp['high'], width, label='High Complexity (Pivots)',
                    color='#444444', hatch='///', edgecolor='#000000', linewidth=0.85)
    
    labels = [COND_LABELS[c] for c in COND_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Diagnostic Resolution Accuracy", fontsize=11)
    ax.set_ylim(0.65, 1.02)
    ax.legend(loc="lower right", frameon=True, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cccccc")
    
    plt.tight_layout()
    save_fig(fig, "fig_subgroup_accuracy.png")
    plt.close()
    print("-> Generated fig_subgroup_accuracy.png (grayscale-safe)")

# -----------------------------------------------------------------------------
# 5. NASA-TLX Radar Chart
# -----------------------------------------------------------------------------
def plot_nasa_tlx():
    categories = ['tlx_mental', 'tlx_physical', 'tlx_temporal', 'tlx_performance', 'tlx_effort', 'tlx_frustration']
    display_cats = ["Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration"]
    
    tlx_means = df_results.groupby('condition')[categories].mean()
    
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    
    # Key conditions with distinct linestyles, markers, and grayscales
    key_conditions = ["control", "bm25", "cma", "medcpt", "graphcare", "gdt", "sdt"]
    radar_styles = {
        "control":   {"ls": ":",  "lw": 1.5, "marker": "o", "ms": 5.5, "color": "#757575"},
        "bm25":      {"ls": "--", "lw": 1.5, "marker": "s", "ms": 5.5, "color": "#686868"},
        "cma":       {"ls": "-.", "lw": 1.6, "marker": "^", "ms": 5.5, "color": "#585858"},
        "medcpt":    {"ls": (0, (3, 1, 1, 1)), "lw": 1.6, "marker": "D", "ms": 5.5, "color": "#4a4a4a"},
        "graphcare": {"ls": (0, (5, 2)), "lw": 1.7, "marker": "v", "ms": 5.5, "color": "#3c3c3c"},
        "gdt":       {"ls": "--", "lw": 2.2, "marker": "p", "ms": 6.5, "color": "#2c2c2c"},
        "sdt":       {"ls": "-",  "lw": 3.0, "marker": "*", "ms": 10.0, "color": "#000000"}
    }
    
    for cond in key_conditions:
        if cond not in tlx_means.index:
            continue
        vals = tlx_means.loc[cond].values.flatten().tolist()
        vals += vals[:1]
        st = radar_styles[cond]
        ax.plot(
            angles, vals,
            linestyle=st["ls"],
            linewidth=st["lw"],
            marker=st["marker"],
            markersize=st["ms"],
            color=st["color"],
            label=COND_LABELS[cond]
        )
        if cond == "sdt":
            ax.fill(angles, vals, color="#000000", alpha=0.12)
            
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_cats, fontsize=10.5)
    ax.set_rlabel_position(30)
    ax.set_ylim(0, 75)
    ax.grid(color="#cccccc", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.15), frameon=True, edgecolor="#cccccc")
    
    plt.tight_layout()
    save_fig(fig, "fig_nasa_tlx_radar.png")
    plt.close()
    print("-> Generated fig_nasa_tlx_radar.png (grayscale-safe)")

# -----------------------------------------------------------------------------
# 6. Queueing Impact Plot
# -----------------------------------------------------------------------------
def plot_queueing():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    
    q_map = {
        "CONTROL": "Control",
        "BM25": "BM25",
        "LLM_RAG": "LLM-RAG",
        "COLBERT": "ColBERT",
        "CMA": "CMA",
        "MEDCPT": "MedCPT",
        "GRAPHCARE": "GraphCare",
        "GDT": "GDT",
        "SDT": "SDT (Ours)"
    }
    
    sys_order = ["CONTROL", "BM25", "LLM_RAG", "COLBERT", "CMA", "MEDCPT", "GRAPHCARE", "GDT", "SDT"]
    cond_sys_map = {
        "CONTROL": "control",
        "BM25": "bm25",
        "LLM_RAG": "llm_rag",
        "COLBERT": "colbert",
        "CMA": "cma",
        "MEDCPT": "medcpt",
        "GRAPHCARE": "graphcare",
        "GDT": "gdt",
        "SDT": "sdt"
    }
    q_sub = df_queue.set_index("System").reindex(sys_order).reset_index()
    
    for i, row in q_sub.iterrows():
        sys_name = row["System"]
        c = cond_sys_map[sys_name]
        st = GRAY_STYLES[c]
        ax.bar(
            i,
            row["W_q (min)"],
            color=st["color"],
            hatch=st["hatch"],
            edgecolor="#000000",
            linewidth=1.2 if c == "sdt" else 0.85
        )
    
    labels = [q_map.get(s, s) for s in q_sub["System"]]
    ax.set_xticks(range(len(q_sub)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Expected ED Queue Wait Time $W_q$ (minutes)", fontsize=11)
    ax.set_ylim(100, 125)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cccccc")
    
    ctrl_wq = q_sub.loc[q_sub["System"] == "CONTROL", "W_q (min)"].values[0]
    ax.axhline(ctrl_wq, color="#333333", linestyle=":", linewidth=1.2, alpha=0.8)
    
    plt.tight_layout()
    save_fig(fig, "fig_queueing_impact.png")
    plt.close()
    print("-> Generated fig_queueing_impact.png (grayscale-safe)")

def main():
    print("Regenerating all 6 publication figures with Grayscale-Safe Design...")
    plot_ttci()
    plot_accuracy()
    plot_cognitive_load()
    plot_subgroup_accuracy()
    plot_nasa_tlx()
    plot_queueing()
    print("All figures successfully regenerated in Grayscale-Safe Design!")

if __name__ == "__main__":
    main()
