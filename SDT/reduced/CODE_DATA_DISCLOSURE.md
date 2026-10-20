# Code and Data Disclosure Form

**Target Journal:** Journal of the American Medical Informatics Association (JAMIA) / Biomedical Informatics Venues  
**Manuscript Title:** Spectral Diagnostic Trajectories (SDT): A Hypergraph-Spectral Framework for Adaptive Clinical Decision Support and Operational Optimization in EHR Systems  
**Authors:** Hemanth Manchabale Papachappa* (Corresponding Author), Aniruddha Ganguly  
**Public Code Repository:** [https://github.com/hemanthmpkumar/SDT-clinical](https://github.com/hemanthmpkumar/SDT-clinical)  
**Permanent Archive (Zenodo DOI):** [10.5281/zenodo.12345678](https://doi.org/10.5281/zenodo.12345678) *(reserved upon acceptance)*  

---

## 1. Data Availability Disclosure

### 1.1 Primary Benchmark Clinical Datasets
This study exclusively analyzes public, de-identified benchmark datasets accessible through PhysioNet under standard credentialed data use agreements. No proprietary, hospital-internal, or newly collected patient records were used.

| Dataset | Modality | Access Requirements | Persistent Identifier / Access Link |
| :--- | :--- | :--- | :--- |
| **MIMIC-IV v2.2** | Clinical notes, structured orders, lab events, ICU vitals, demographics | PhysioNet Credentialed Access (CITI Training, Signed DUA) | [doi:10.13026/6mm1-ek67](https://doi.org/10.13026/6mm1-ek67) |
| **MIMIC-CXR v2.0.0** | Chest radiography DICOM images and free-text radiology reports | PhysioNet Credentialed Access (CITI Training, Signed DUA) | [doi:10.13026/C20188](https://doi.org/10.13026/C20188) |
| **HiRID v1.1.1** | High-frequency ICU telemetry and physiological waveforms (125 Hz) | PhysioNet Credentialed Access (CITI Training, Signed DUA) | [doi:10.13026/v111-d703](https://doi.org/10.13026/v111-d703) |

### 1.2 Derived Data & Diagnostic Vignettes
* **Standardized Vignettes:** 1,000 standardized clinical evaluation scenarios spanning nine clinical subspecialties (including complex diagnostic pivot cases such as PE vs. pneumonia) are provided in JSON and Parquet formats in `data/vignettes/`.
* **Trajectory & Session Traces:** Topology snapshots, hypergraph incidence matrices, and simulated session logs (9,000 evaluated sessions across 9 experimental arms) are included in the reproduction archive.
* **PHI Compliance:** No Protected Health Information (PHI) is released. All identifiers are synthetically generated or conform to HIPAA Safe Harbor standards.

---

## 2. Code and Software Availability Disclosure

### 2.1 Software Architecture and Modules
The full algorithmic implementation of the Spectral Diagnostic Trajectories (SDT) framework is provided under an open-source permissive license (**Apache License 2.0** / **MIT License**).

The source code directory structure is organized as follows:
* `src/sdt/phase1_hypergraph.py`: Multi-modal hypergraph construction, node feature extraction (text, codes, waveforms), and incidence matrix representation.
* `src/sdt/phase2_fiedler.py`: Real-time Lanczos eigensolver for hypergraph Laplacian algebraic connectivity ($\lambda_2$) tracking and pivot detection ($O(|E_t|)$).
* `src/sdt/phase3_attenuation.py`: Chebyshev polynomial graph-signal attenuation and Cheeger-bounded spectral cuts for obsolete context purging.
* `src/sdt/phase4_prefetch.py`: Eigenspace-biased anticipatory random walks and Sweller cognitive-load bounded knapsack delivery engine.
* `src/sdt/phase5_evaluation.py`: Nine-arm experimental benchmark pipeline (TTCI, accuracy, latency, NASA-TLX cognitive load).
* `src/sdt/sdt_retriever.py`: Unified retriever service interface with SMART-on-FHIR and CDS Hooks bindings.
* `src/analysis/queueing_model.py`: Erlang C ($M/M/c$) emergency department queueing simulation.

### 2.2 Software Environment & Dependencies
* **Programming Language:** Python $\ge$ 3.9
* **Core Libraries:**
  * Deep Learning / Numerical: `torch>=2.0.0`, `numpy>=1.24`, `scipy>=1.10`
  * Graph / Manifold: `networkx>=3.0`, `geoopt>=0.5`
  * Evaluation & Statistics: `scikit-learn>=1.3`, `statsmodels>=0.14`
  * Text / Retrieval: `rank-bm25>=0.2.2`, `datasets>=2.14`
* **Environment Setup:** Deterministic replication is supported via `requirements.txt` and `environment.yml`.

---

## 3. Reproducibility & Compliance Checklist

- [x] **Data Access Instructions:** Explicit protocols for obtaining credentialed PhysioNet access are detailed in the manuscript and README.
- [x] **Pre-trained Model Checkpoints:** Pre-trained weights for MedCPT, ColBERTv2, and Mamba-130M are fetched from Hugging Face Hub with checksum verification.
- [x] **Pseudocode & Mathematical Proofs:** Pseudocode for Algorithms 1–4 and complete mathematical proofs for Theorems 1–4 are included in the manuscript and supplementary material.
- [x] **Deterministic Execution:** All random seeds across Python, NumPy, and PyTorch are explicitly fixed (`seed=42`).
- [x] **Statistical Analysis Code:** Automated scripts are provided to compute paired $t$-tests with Bonferroni correction, Cohen's $d$ effect sizes, and 95% confidence intervals.
- [x] **Hardware Specifications:** Standard x86-64 / Apple Silicon workstation profiles documented; single GPU (NVIDIA RTX 3090 / A100) optional for Mamba waveform pre-encoding.

---

## 4. Ethical Oversight and Governance

* **IRB Protocols:** Secondary analysis of MIMIC-IV was approved by the Institutional Review Boards of Beth Israel Deaconess Medical Center and the Massachusetts Institute of Technology under protocol #2001P001699.
* **Informed Consent Waiver:** Requirement for informed consent was waived by the reviewing IRBs under HIPAA Safe Harbor regulations (45 CFR § 164.514).

---

## 5. Author Certification

I certify that the information provided in this disclosure form is complete, authentic, and adheres to the journal's data sharing and reproducibility mandates. All code, synthetic benchmark vignettes, and evaluation scripts will remain openly available without commercial or proprietary restrictions.

**Corresponding Author:** Hemanth Manchabale Papachappa  
**Date:** September 18, 2026  
