import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_bar_with_errors(summary_df, metric_col, err_col, title, ylabel, filename):
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    # Custom color palette for the 5 arms
    colors = {"control": "#95a5a6", "bm25": "#7f8c8d", "cma": "#3498db", "gdt": "#9b59b6", "sdt": "#e74c3c"}
    
    ax = sns.barplot(
        data=summary_df, 
        x="condition", 
        y=metric_col,
        palette=colors,
        order=["control", "bm25", "cma", "gdt", "sdt"]
    )
    
    # Add error bars manually since we have standard errors in summary_df
    x_coords = [p.get_x() + p.get_width()/2 for p in ax.patches]
    y_coords = summary_df.set_index("condition").loc[["control", "bm25", "cma", "gdt", "sdt"], metric_col].values
    y_errs = summary_df.set_index("condition").loc[["control", "bm25", "cma", "gdt", "sdt"], err_col].values
    
    plt.errorbar(x_coords, y_coords, yerr=y_errs, fmt="none", c="black", capsize=5)
    
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Retriever Condition", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_tlx_radar(results_df, filename):
    tlx_cols = ['tlx_mental', 'tlx_physical', 'tlx_temporal', 'tlx_performance', 'tlx_effort', 'tlx_frustration']
    means = results_df.groupby('condition')[tlx_cols].mean().reindex(["control", "bm25", "cma", "gdt", "sdt"])
    
    # Spider plot setup
    labels = ["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"]
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = {"control": "#95a5a6", "bm25": "#7f8c8d", "cma": "#3498db", "gdt": "#9b59b6", "sdt": "#e74c3c"}
    
    for condition, color in colors.items():
        if condition not in means.index:
            continue
        values = means.loc[condition].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=condition.upper())
        ax.fill(angles, values, color=color, alpha=0.1)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.title("NASA-TLX Workload Profiles", y=1.08, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    out_dir = Path("outputs")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Load Data
    print("Loading results...")
    summary_df = pd.read_csv(out_dir / "sdt_summary.csv")
    results_df = pd.read_csv(out_dir / "results.csv")
    
    print("Generating Time to Information Plot...")
    plot_bar_with_errors(
        summary_df, "time_to_info_mean", "time_to_info_se",
        "Time to Information (Seconds)", "Mean Time (s)",
        plots_dir / "time_to_info.png"
    )
    
    print("Generating Cognitive Load Plot...")
    plot_bar_with_errors(
        summary_df, "cognitive_load_mean", "cognitive_load_se",
        "Overall Cognitive Load", "Mean Cognitive Load",
        plots_dir / "cognitive_load.png"
    )
    
    print("Generating Diagnostic Accuracy Plot...")
    plot_bar_with_errors(
        summary_df, "accuracy_mean", "accuracy_se",
        "Diagnostic Accuracy (Resolution Rate)", "Mean Accuracy",
        plots_dir / "accuracy.png"
    )
    
    print("Generating NASA-TLX Radar Chart...")
    plot_tlx_radar(results_df, plots_dir / "nasa_tlx_radar.png")
    
    print(f"All plots saved to {plots_dir}")

if __name__ == "__main__":
    main()
