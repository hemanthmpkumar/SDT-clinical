# Spectral Trajectories Research Workspace

This repository houses the mathematical and computational frameworks extending spectral graph theory and dynamic hypergraph Laplacians into high-dimensional streaming environments.

---

## Workspace Structure

```
.
├── .venv/                   # Shared Python virtual environment
├── .gitignore               # Workspace-wide Git exclusion rules
├── SDT/                     # Spectral Diagnostic Trajectories (Clinical Decision Support)
│   ├── README.md            # Detailed SDT documentation and phase descriptions
│   ├── requirements.txt     # Python dependencies for SDT
│   ├── run_experiments.py   # One-shot reproduction script
│   ├── src/                 # 5-Phase SDT architecture & simulation harness
│   ├── tests/               # Unit and regression test suite
│   ├── data/                # Clinical vignettes and processed corpus
│   ├── latex/               # Manuscript sources (JAMIA, NPJ Digital Medicine, IJDS)
│   └── outputs/             # Simulation results, plots, and queueing tables
└── SCT/                     # Spectral Cosmic Trajectories (Planetary & Astrophysical Discovery)
    └── (In Development)     # Multi-modal hypergraph framework for streaming space data
```

---

## Quick Start

### 1. Environment Setup

Activate the workspace virtual environment:

```bash
# On macOS / Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

### 2. Clinical SDT Subsystem

To work with or run the clinical decision support experiments:

```bash
# Navigate to SDT
cd SDT

# Install dependencies if not already installed
pip install -r requirements.txt

# Run the 5-arm clinical search crossover simulation
python src/experiments/run.py

# Or run the full reproduction pipeline (data splits, model training, experiments, plots)
python run_experiments.py
```

To run tests across SDT from the root directory:

```bash
PYTHONPATH=SDT pytest SDT/tests/
```

### 3. Planetary & Astrophysical SCT Subsystem

For the upcoming **Spectral Cosmic Trajectories (SCT)** framework:
- See the dedicated directory at [`SCT/`](SCT/) for the adaptive scientific discovery proposal and hypergraph Laplacian implementation on multi-messenger/planetary data streams (JWST spectra, light curves, retrieval grids).

---

## Documentation Links

- [SDT Architecture & Framework Documentation](SDT/README.md)
- [SDT 5-Phase Implementation](SDT/src/sdt/)
- [SCT Project Directory](SCT/)
