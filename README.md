# Spectral Diagnostic Trajectories (SDT)

This repository contains the official implementation of the **Spectral Diagnostic Trajectories (SDT)** framework for clinical search, as well as the experimental harness for evaluating simulated clinician chart review on MIMIC-IV vignettes.

The SDT framework is a radical evolution over prior session-based search methods. It models diagnostic hypotheses not as dense vectors in Euclidean space, but as an evolving **multi-modal hypergraph** where the topology directly encodes the clinical state.

## Core Architecture (Phase 1–5)

The code is strictly organized around the 5 mathematical phases of the SDT framework, all contained within `src/sdt/`:

- **Phase 1 (`phase1_hypergraph.py`)**: Defines the `MultiModalHypergraph` state space. Replaces flat matrices with dynamic text, EHR, and physiological nodes connected by composite temporal-semantic weights (Zhou et al. adjacency formulation).
- **Phase 2 (`phase2_fiedler.py`)**: The `SpectralPivotDetector`. Tracks the Fiedler value $\lambda_2(L_t)$ of the normalized graph Laplacian. Detects abrupt topic pivots via the interference gate trigger $\Delta\lambda_2 > \tau_{pivot}$.
- **Phase 3 (`phase3_attenuation.py`)**: The `GraphSignalAttenuator`. Upon interference, applies a high-pass graph signal filter and partitions the active window into $V_{stale}$ and $V_{active}$ via the Fiedler vector cut, generating the clean retrieval seed vector.
- **Phase 4 (`phase4_prefetch.py`)**: The `AnticipatoryPrefetchEngine`. Executes spectrally-biased random walks over the global corpus graph. Bounds candidates via a greedy knapsack constraint matching Sweller’s Cognitive Load limit.
- **Phase 5 (`phase5_evaluation.py`)**: Evaluates TTCI, MRR, Contextual Precision, and maps the simulated latency reductions into Emergency Department waiting room improvements via a closed-form Erlang C queueing model (M/M/c).

`sdt_retriever.py` orchestrates these 5 phases and exposes an interface compatible with standard sparse and dense search pipelines.

## Installation

We recommend using a Python virtual environment to manage dependencies:

```bash
# Create a new virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

*Note: For performance, most Laplacian eigendecompositions in the SDT pipeline use `scipy.sparse.linalg` directly, without requiring `networkx` conversion overhead.*

## Running the Experiments

To run the simulated cross-over trial comparing SDT against the TF-IDF (Control) and BM25 sparse baselines:

```bash
python src/experiments/run.py
```

This will run the full simulation and save outputs to the `outputs/` directory:
- `sdt_results.csv`: Raw session metrics (time-to-info, accuracy, cognitive load)
- `sdt_summary.csv`: Aggregated means and standard errors per condition
- `sdt_queueing.csv`: The Erlang C operational impact projection table (mapping Time-To-Correct-Information to real ED wait-time improvements).

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/
│   └── processed/          # Simulated patient vignettes and corpus
├── src/
│   ├── sdt/                # The 5-Phase SDT Architecture
│   │   ├── sdt_retriever.py
│   │   ├── phase1_hypergraph.py
│   │   ├── phase2_fiedler.py
│   │   ├── phase3_attenuation.py
│   │   ├── phase4_prefetch.py
│   │   └── phase5_evaluation.py
│   ├── models/             # Baseline retrievers (BM25, Base TF-IDF)
│   ├── experiments/        # Simulation and crossover testing scripts
│   │   ├── run.py
│   │   └── simulate_users.py
│   └── analysis/           # Post-experiment analysis & plotting
└── outputs/                # Experiment metrics and queueing tables
```
