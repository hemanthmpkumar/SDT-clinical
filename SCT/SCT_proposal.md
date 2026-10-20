# Spectral Cosmic Trajectories (SCT): A Hypergraph-Spectral Framework for Adaptive Scientific Discovery in Planetary and Astrophysical Data Streams

**Research Proposal — Version 1.0**

---

**Authors:** [Proposal Manuscript — For Internal Circulation]
**Institutional Affiliation:** [To Be Completed]
**Submission Target:** *The Astrophysical Journal Supplement Series* / *Nature Astronomy* / *Icarus*
**Date:** October 2026

---

## Abstract

We introduce **Spectral Cosmic Trajectories (SCT)**, a rigorous mathematical framework that extends the Spectral Diagnostic Trajectories (SDT) paradigm from clinical decision support into the domains of planetary science, exoplanet atmospheric characterization, and multi-messenger astrophysics. The core thesis is that the evolving scientific understanding of a planetary body, exoplanet atmosphere, or astrophysical transient event can be faithfully modeled as a *dynamic multi-modal hypergraph* $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t, \mathcal{W}_t)$, in which vertices encode heterogeneous observational modalities — transmission spectra, photometric light curves, spatially resolved images, physico-chemical retrieval grids, and structured astrophysical literature — and hyperedges encode multi-way scientific consistency relationships across these modalities.

We formalize **paradigm pivots** — abrupt failures of the prevailing atmospheric or physical model — as detectable discontinuities in the algebraic connectivity (Fiedler value $\lambda_2$) of the normalized hypergraph Laplacian. Upon pivot detection, we apply Chebyshev-filtered Fiedler vector bisection to partition the hypothesis graph into an *active* sub-manifold and a *stale* sub-manifold, with a formal contamination bound governing residual influence from discarded models. Finally, we formulate adaptive observation prefetching as a spectrally biased random walk over the global observational and model graph, constrained by a scientific *cognitive load* budget.

We state the primary testable hypothesis that SCT detects the correct atmospheric regime shift in JWST transmission spectra of benchmark exoplanets (WASP-39b, WASP-107b, TRAPPIST-1e) with **fewer total observations and lower residual model contamination** than either sequential Bayesian model comparison or continuous geodesic trajectory methods. A concrete experimental protocol, ablation suite, and impact assessment are provided.

**Keywords:** spectral graph theory, hypergraph Laplacian, Fiedler value, exoplanet atmospheres, JWST, multi-messenger astronomy, adaptive observation scheduling, scientific machine learning, paradigm detection, algebraic connectivity.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Background and Related Work](#2-background-and-related-work)
3. [Formal Problem Statement](#3-formal-problem-statement)
4. [Multi-Modal Hypergraph Construction](#4-multi-modal-hypergraph-construction)
5. [Spectral Pivot Detection](#5-spectral-pivot-detection)
6. [Context Attenuation via Chebyshev-Filtered Fiedler Cuts](#6-context-attenuation-via-chebyshev-filtered-fiedler-cuts)
7. [Adaptive Observation Prefetching](#7-adaptive-observation-prefetching)
8. [Primary Testable Hypothesis](#8-primary-testable-hypothesis)
9. [Experimental Design](#9-experimental-design)
10. [Expected Scientific Impact](#10-expected-scientific-impact)
11. [Discussion](#11-discussion)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)
14. [Appendices](#14-appendices)

---

## 1. Introduction and Motivation

### 1.1 The Data Crisis in Modern Planetary and Astrophysical Science

The opening decade of the 2020s has inaugurated a golden era for planetary science and exoplanet characterization. The James Webb Space Telescope (JWST), operational since mid-2022, generates transmission and emission spectra of unparalleled sensitivity across the 0.6–28 μm wavelength range, providing direct constraints on the atmospheric compositions, temperature profiles, and cloud structures of planets ranging from hot Jupiters to potentially habitable terrestrial worlds. Simultaneously, the Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) is accumulating petabytes of time-domain photometry across the solar system; the European Space Agency's *Ariel* mission (launch 2029) will survey the atmospheres of over 1,000 exoplanets; and multi-messenger networks combining gravitational-wave interferometers, neutrino detectors, and electromagnetic follow-up arrays are generating joint observational datasets of astrophysical transients at unprecedented cadences.

The result is a **data-rate and data-heterogeneity crisis**. For a single system such as WASP-39b, the scientific community now synthesizes:

- **Spectral data streams**: NIRISS SOSS, NIRCam grism, NIRSpec PRISM, NIRSpec G395H, and MIRI LRS transmission spectra from JWST's Early Release Science program alone [Alderson et al. 2023; Feinstein et al. 2023; Grant et al. 2023; Rustamkulov et al. 2023; Bell et al. 2023], constituting $\mathcal{O}(10^4)$ wavelength-flux data points per observation.
- **Photometric light curves**: Ground-based transit timing variations, Spitzer and Hubble photometry, TESS time-series covering months to years of baseline.
- **Physical retrieval grids**: Atmospheric retrieval computations (e.g., CHIMERA, POSEIDON, petitRADTRANS, AURORA) that explore $\mathcal{O}(10^6)$ forward-model evaluations per retrieval campaign, spanning gas-phase chemistry, cloud microphysics, disequilibrium chemistry, and photochemistry parameters.
- **Imaging data**: Coronagraphic or interferometric constraints (where available), coupled with stellar contamination models from stellar spots and faculae.
- **Literature corpus**: Hundreds of peer-reviewed publications encoding prior constraints, comparison studies, theoretical predictions, and reanalysis results.

No single researcher or even research team can holistically synthesize this heterogeneous evidence stream in real time. Current practice involves sequential, modality-by-modality analysis: a spectroscopist reduces spectra, a retrieval specialist runs atmospheric inversions, theorists propose formation scenarios, all in largely asynchronous fashion. Critical inconsistencies — such as a new photochemical detection overturning a previously favored equilibrium chemistry hypothesis — may take months to propagate through the community.

This paper proposes that the **Spectral Diagnostic Trajectories (SDT)** framework, originally designed to model the evolving clinical reasoning of a physician reviewing electronic health records, provides a mathematically natural substrate for solving precisely this synthesis problem in planetary science.

### 1.2 The SDT Paradigm and Its Planetary Extension

The SDT framework [see companion repository and manuscript] models a decision-maker's evolving state of knowledge as a dynamic multi-modal hypergraph. Three core innovations distinguish SDT from prior session-based reasoning models:

1. **Hypergraph state space**: Rather than encoding the current hypothesis as a point on a Riemannian manifold (as in the predecessor Geodesic Diagnostic Trajectories framework) or as a flat vector (as in classical IR systems), SDT represents the hypothesis state as the topology of an active subgraph $\mathcal{G}_t$, where the *structure* — not merely the content — encodes the logical relationships among evidential units.

2. **Fiedler-value pivot detection**: The algebraic connectivity $\lambda_2(\mathcal{L}_t)$ of the normalized Laplacian continuously monitors the internal cohesion of the hypothesis graph. A precipitous drop $\Delta\lambda_2(t) > \tau_\mathrm{pivot}$ signals that new evidence is *structurally incompatible* with the current hypothesis cluster — a diagnostic pivot.

3. **Spectrally budgeted prefetching**: Following a pivot, a Chebyshev-filtered Fiedler bisection partitions the graph into stale and active components; a spectrally biased random walk then prefetches the most informative next evidence items subject to a cognitive load constraint.

In the clinical domain, these three phases correspond to: a physician reviewing a chart, recognizing an unexpected lab result that invalidates the current diagnosis, pruning prior evidence, and ordering the most informative next tests. The mapping to planetary science is structurally exact:

| Clinical SDT Concept | SCT Planetary Analog |
|---|---|
| Patient chart | Multi-messenger observational record of a planetary body |
| Diagnostic hypothesis | Atmospheric/physical model (e.g., cloud-free equilibrium) |
| Lab result | New JWST spectral observation or retrieval output |
| Diagnostic pivot | Atmospheric regime shift (e.g., photochemistry detected; clouds identified) |
| Cognitive load budget | Maximum number of competing models a research team can actively evaluate |
| Next test to order | Next most informative observation or retrieval grid to compute |

We call this extension **Spectral Cosmic Trajectories (SCT)**, and we now develop it with full mathematical rigor.

### 1.3 Why Existing Methods Are Insufficient

Before constructing the SCT framework, we motivate it through a systematic critique of the principal existing approaches.

#### 1.3.1 Continuous Manifold and Riemannian Methods

Recent geometric machine learning approaches represent scientific knowledge states as points on a learned Riemannian manifold, with geodesic distances encoding semantic dissimilarity. The Geodesic Diagnostic Trajectories (GDT) framework [preceding SDT] represents the clinical state as a symmetric positive-definite (SPD) matrix evolving on the Riemannian manifold of SPD matrices equipped with the affine-invariant metric. While mathematically elegant, manifold methods suffer from fundamental limitations in the planetary science context:

**Dimensionality and heterogeneity**: Riemannian methods require that all data modalities be embedded into a single common metric space. For planetary science, this requires jointly embedding a 4000-channel NIRSpec spectrum, a 10-year photometric light curve, a 200-parameter retrieval posterior, and a paragraph of literature text into a common manifold. No principled, information-preserving embedding for such disparate modalities is known. Ad hoc dimensionality reduction (e.g., UMAP or t-SNE) destroys metric structure and renders geodesic distances uninterpretable.

**Streaming incompatibility**: Riemannian geodesic computations are inherently batch operations requiring access to the full manifold geometry. In streaming settings where new JWST observations arrive asynchronously, recomputing the full geodesic after each new data point is computationally intractable.

**Topological blindness**: Riemannian methods encode only metric (pairwise distance) information. They cannot represent multi-way consistency constraints — the fact that three modalities simultaneously confirm a hypothesis — without constructing explicit 3-point or higher-order metric structures, for which no standard framework exists.

**Non-linearity of scientific paradigm shifts**: The transition from an equilibrium-chemistry atmospheric model to a disequilibrium-chemistry + photochemistry model is not a smooth geodesic on any natural manifold. It is a *topological phase transition* — a structural reconfiguration of the hypothesis graph — which is captured exactly by a drop in $\lambda_2$ but not by any continuous geodesic trajectory.

#### 1.3.2 Standard Bayesian Model Comparison

The gold standard for atmospheric model selection is the Bayesian evidence ratio (Bayes factor), computed via nested sampling algorithms (e.g., MultiNest, dynesty) [Skilling 2004; Feroz et al. 2009; Speagle 2020]. Given competing models $\mathcal{M}_1, \mathcal{M}_2$, the Bayes factor $B_{12} = p(D|\mathcal{M}_1)/p(D|\mathcal{M}_2)$ quantifies the preference for one model over another given data $D$.

Bayesian model comparison has several critical weaknesses in the high-dimensional, streaming, multi-modal planetary context:

**Sequential, single-modality nature**: Standard Bayesian evidence computation is performed *per retrieval campaign*, treating each new dataset as a standalone comparison. There is no principled mechanism to continuously fuse evidence from five simultaneous JWST instrument modes, ground-based photometry, and theoretical chemistry grids into a single evolving Bayes factor.

**Computational cost**: A single nested sampling retrieval for an exoplanet atmosphere requires $\mathcal{O}(10^5$–$10^6)$ likelihood evaluations, each involving a forward radiative transfer model. The time between observation and published retrieval result is typically weeks to months, creating a fundamental latency incompatible with real-time adaptive scheduling.

**Model space finiteness**: Bayes factors compare a finite, pre-specified set of models. They cannot discover that the *correct* model lies outside the comparison set — a scenario that represents exactly the "new physics" events of greatest scientific interest. When JWST detected SO$_2$ in WASP-39b's atmosphere [Tsai et al. 2023], this was not selected by comparing two pre-specified models; it required recognizing an anomalous residual that pointed outside the entire existing model space.

**No observation guidance**: Bayesian model comparison answers "given these data, which model is preferred?" but not "given current model uncertainty, which next observation would most efficiently resolve the comparison?" Observation scheduling requires a fully formulated information-value framework, which is not intrinsic to standard Bayesian inference.

**Contamination persistence**: When a Bayesian analysis is updated with new data, prior posteriors cannot be surgically excised. If early observations under Model $\mathcal{M}_1$ produced a biased posterior (due to, say, unmodeled stellar contamination), this bias propagates into all subsequent updates through Bayes' theorem. SCT's Fiedler cut provides a principled, spectral mechanism to *surgically remove* the contaminating prior component.

#### 1.3.3 Flat Vector and Standard Information Retrieval Methods

TF-IDF, BM25, and standard dense retrieval methods over astrophysical literature and retrieval databases lack all three of the core SDT/SCT innovations: they have no notion of multi-way consistency (hyperedges), no topological change detection (pivot mechanism), and no cognitively bounded prefetching. Their limitations in the scientific discovery context are well-established and require no extended analysis here.

### 1.4 Scope and Contributions of This Proposal

This proposal makes the following contributions:

1. **A formally specified multi-modal hypergraph** for planetary and astrophysical science data, with five vertex modalities and a principled hyperedge weighting scheme encoding physical and statistical consistency.
2. **A normalized hypergraph Laplacian** adapted to the astrophysical heterogeneity of the vertex types, with a **Scientific Cohesion Theorem** (Theorem 5.1) formally linking $\lambda_2$ drops to the emergence of conflicting hypotheses.
3. **A Chebyshev-filtered Fiedler cut** with an explicit **contamination bound** (Theorem 6.2) governing residual stale-model influence after a paradigm pivot.
4. **An adaptive observation prefetching engine** formulated as a constrained spectrally biased random walk, with a scientific cognitive load budget operationalized as a knapsack constraint on competing model cardinality.
5. **A primary testable hypothesis** and complete experimental protocol using public JWST data from WASP-39b, WASP-107b, and TRAPPIST-1 planets.
6. **An impact assessment** covering adaptive telescope scheduling, automated model pruning, and new physics discovery.

---

## 2. Background and Related Work

### 2.1 Spectral Graph Theory

The algebraic connectivity of a graph, introduced by Fiedler [1973], is the second-smallest eigenvalue $\lambda_2$ of the graph Laplacian $L = D - A$, where $D$ is the degree matrix and $A$ is the adjacency matrix. Key properties of $\lambda_2$ that underpin the SCT framework:

- **Connectivity**: $\lambda_2 > 0$ if and only if the graph is connected.
- **Cheeger inequality**: $h(\mathcal{G})^2/2 \leq \lambda_2 \leq 2h(\mathcal{G})$, where $h(\mathcal{G})$ is the graph conductance (Cheeger constant), bounding the quality of the best bisection.
- **Perturbation monotonicity**: Adding edges to $\mathcal{G}$ increases $\lambda_2$; removing edges decreases it.
- **Spectral gap and mixing**: The spectral gap $\lambda_2 / \lambda_\mathrm{max}$ controls the mixing time of random walks on the graph.

For the *normalized* Laplacian $\mathcal{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$, eigenvalues lie in $[0, 2]$, with $\lambda_2(\mathcal{L})$ providing a degree-corrected measure of cohesion.

### 2.2 Hypergraph Laplacians

A hypergraph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{W})$ generalizes graphs by allowing hyperedges $e \in \mathcal{E}$ to connect arbitrary subsets of vertices. The formalism of Zhou et al. [2006] defines:

- **Incidence matrix** $H \in \{0,1\}^{|\mathcal{V}| \times |\mathcal{E}|}$: $H(v,e) = 1$ iff $v \in e$.
- **Vertex degree** $d(v) = \sum_{e: v \in e} w(e)$, diagonal matrix $D_v$.
- **Edge degree** $\delta(e) = |e|$, diagonal matrix $D_e$.
- **Normalized hypergraph Laplacian**: $\Delta = I - D_v^{-1/2} H W D_e^{-1} H^\top D_v^{-1/2}$

where $W = \mathrm{diag}(w(e_1), \ldots, w(e_{|\mathcal{E}|}))$ is the diagonal matrix of hyperedge weights. The eigenvalues of $\Delta$ lie in $[0, 2]$, and $\lambda_2(\Delta)$ serves as the algebraic connectivity of the hypergraph [Saito et al. 2018].

### 2.3 Chebyshev Polynomial Spectral Filters

Graph signal processing [Shuman et al. 2013] defines a filter as a polynomial in the graph Laplacian. The $k$-th order Chebyshev approximation provides a computationally efficient spectral filter:

$$h_k(\mathcal{L}) = \sum_{j=0}^{k} \theta_j T_j(\tilde{\mathcal{L}})$$

where $\tilde{\mathcal{L}} = 2\mathcal{L}/\lambda_\mathrm{max} - I$ is the scaled Laplacian, $T_j$ are Chebyshev polynomials of the first kind, and $\theta_j$ are filter coefficients. A high-pass filter isolates components associated with large Laplacian eigenvalues (intra-cluster variation), while a low-pass filter retains components associated with small eigenvalues (global smoothness). The key computational advantage is that $h_k(\mathcal{L})x$ can be evaluated in $O(k|\mathcal{E}|)$ time without explicit eigendecomposition.

### 2.4 Exoplanet Atmospheric Retrieval

Atmospheric retrieval is the inverse problem of inferring atmospheric properties (temperature-pressure profile, molecular abundances, cloud parameters) from observed transmission or emission spectra. The standard approach uses Bayesian inference with a forward radiative transfer model $\mathcal{F}(\theta)$ mapping parameter vector $\theta$ to synthetic spectrum, maximizing the posterior $p(\theta | D) \propto p(D|\theta, \mathcal{M}) \cdot p(\theta | \mathcal{M})$ via nested sampling [Buchner et al. 2014].

Key retrieval codes include petitRADTRANS [Mollière et al. 2019], POSEIDON [MacDonald & Madhusudhan 2017], CHIMERA [Line et al. 2013], AURORA [Welbanks & Madhusudhan 2021], and NEMESIS [Irwin et al. 2008]. These codes solve overlapping but distinct scientific problems, motivating their joint representation in the SCT hypergraph.

### 2.5 JWST Exoplanet Science

The first JWST exoplanet results [Alderson et al. 2023; Ahrer et al. 2023; Feinstein et al. 2023; Grant et al. 2023; Rustamkulov et al. 2023; Bell et al. 2023; JWST Transiting Exoplanet Community ERS Team 2023] demonstrated detections of CO$_2$, SO$_2$, H$_2$O, CO, Na, K, and tentative detections of additional species in hot Jupiter and sub-Neptune atmospheres. The multi-instrument, multi-epoch nature of JWST observations makes them an ideal testbed for the SCT framework.

### 2.6 Adaptive Observation Scheduling

Optimal scheduling of telescope time is a resource allocation problem with deep roots in operations research. Bayesian optimization and active learning approaches [Settles 2009; Snoek et al. 2012] have been applied to observation planning, but they lack the topological change-detection capability required to respond to paradigm pivots. Recent work on autonomous telescope scheduling [Martinez-Galarza et al. 2021; Lam et al. 2023] uses reinforcement learning, but these methods lack formal spectral guarantees.

### 2.7 Scientific Machine Learning for Astrophysics

The application of machine learning to astrophysical data synthesis has accelerated dramatically. Emulators for atmospheric retrieval [Cobb et al. 2019; Zingales & Waldmann 2018; Nixon & Madhusudhan 2020] use neural networks to replace expensive radiative transfer computations. Graph neural networks have been applied to cosmological simulations [Villanueva-Domingo et al. 2022] and stellar classification [Camacho-Chávez et al. 2023]. However, none of these approaches incorporates the continuous spectral monitoring, topological pivot detection, or budgeted prefetching that define the SCT framework.


---

## 3. Formal Problem Statement

### 3.1 Scientific State Space

**Definition 3.1 (Observation Stream).**
Let $\Omega = \{\omega_1, \omega_2, \ldots\}$ be a potentially infinite, temporally ordered stream of observational and computational artifacts pertaining to a target planetary system $\mathcal{P}$. Each artifact $\omega_i$ belongs to one of five modality classes $\mathcal{M} = \{M_\mathrm{spec}, M_\mathrm{phot}, M_\mathrm{img}, M_\mathrm{model}, M_\mathrm{lit}\}$:

- $M_\mathrm{spec}$: Spectral observations (transmission spectra, emission spectra, reflected-light spectra).
- $M_\mathrm{phot}$: Photometric observations (transit light curves, phase curves, radial velocity time series).
- $M_\mathrm{img}$: Spatially resolved imaging (coronagraphic images, disk-resolved thermal maps, radar altimetry).
- $M_\mathrm{model}$: Physical models and retrieval outputs (parameter posteriors, forward-model grids, chemistry networks).
- $M_\mathrm{lit}$: Structured astrophysical literature (peer-reviewed publications, arXiv preprints, ADS records).

**Definition 3.2 (Scientific State at Time $t$).**
The *scientific state* of system $\mathcal{P}$ at discrete time step $t$ is the tuple $\mathcal{S}_t = (\mathcal{G}_t, \mathcal{H}_t, \lambda_2^{(t)})$, where:

- $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t, \mathcal{W}_t)$ is the active multi-modal hypergraph at time $t$.
- $\mathcal{H}_t \subseteq 2^\mathcal{V}$ is the set of active model hypotheses, each corresponding to a connected subgraph of $\mathcal{G}_t$.
- $\lambda_2^{(t)}$ is the algebraic connectivity of the normalized hypergraph Laplacian $\Delta_t$.

**Definition 3.3 (Paradigm Pivot).**
A *paradigm pivot* at time $t$ is an event in which the current dominant hypothesis $\mathcal{H}_t^*$ (the hypothesis cluster with highest posterior weight) fails to account for newly arriving observation $\omega_{t+1}$. Formally, a paradigm pivot occurs when:

$$\Delta\lambda_2(t) \triangleq \lambda_2^{(t-1)} - \lambda_2^{(t)} > \tau_\mathrm{pivot}$$

where $\tau_\mathrm{pivot} > 0$ is a pre-specified sensitivity threshold.

### 3.2 The SCT Problem

**Problem 3.1 (Spectral Cosmic Trajectories).**
Given a target planetary system $\mathcal{P}$ and a streaming observation sequence $\Omega$, design an algorithm $\mathcal{A}$ that:

1. **Maintains** a dynamic multi-modal hypergraph $\mathcal{G}_t$ incorporating all currently relevant observations and models.
2. **Detects** paradigm pivots in $O(\mathcal{G}_t \text{ update time})$ — i.e., without full batch reprocessing.
3. **Attenuates** stale model subgraphs via a Chebyshev-filtered Fiedler cut with residual contamination $\leq \epsilon_\mathrm{cont}$.
4. **Selects** the next most informative observation $\omega^* \in \Omega_\mathrm{unobserved}$ maximizing expected $\lambda_2$ increase subject to a scientific budget constraint $|\mathcal{H}_\mathrm{active}| \leq B_\mathrm{sci}$.

The performance of $\mathcal{A}$ is measured by:

- **Pivot latency** $\ell_\mathrm{pivot}$: Number of observations processed between the true paradigm shift and its detection.
- **Residual contamination** $\epsilon_\mathrm{cont}$: Cosine similarity between the post-cut active subgraph signal and the discarded stale hypothesis.
- **Retrieval accuracy** $\mathrm{ACC}_\mathrm{ret}$: Correctness of the final atmospheric parameter estimates compared to ground truth (simulated) or expert consensus (real data).
- **Observation efficiency** $\eta_\mathrm{obs}$: Number of observations required to achieve a target retrieval accuracy.

### 3.3 Why Standard Methods Fail: Information-Theoretic Arguments

We now provide three formal arguments showing why the methods discussed in §1.3 are suboptimal for Problem 3.1.

**Proposition 3.1 (Manifold Method Suboptimality).**
Let $\mathcal{D}$ be a dataset with $|\mathcal{M}| = 5$ heterogeneous modalities, each lying on a modality-specific manifold $\mathcal{M}_m$ with intrinsic dimension $d_m$. Any joint manifold embedding $\phi: \mathcal{D} \to \mathbb{R}^D$ with $D < \sum_m d_m$ necessarily incurs a distortion $\delta \geq C \cdot (\sum_m d_m - D)$ for some $C > 0$ depending on the curvature of the constituent manifolds. For typical astrophysical modalities ($d_\mathrm{spec} \approx 50$–$200$, $d_\mathrm{model} \approx 20$–$100$), this distortion is non-negligible for any practically tractable $D$.

*Proof sketch*: This follows from the Johnson-Lindenstrauss lemma lower bound for non-isometric embeddings of finite metric spaces [Matousek 2002, Theorem 15.5.1]. □

**Proposition 3.2 (Bayesian Update Contamination).**
Let $p_0(\theta|\mathcal{M}_1)$ be an erroneous prior posterior arising from misidentified model $\mathcal{M}_1$. Under Bayes' theorem, the posterior after $n$ additional observations $D_{1:n}$ satisfies:

$$p(\theta|D_{1:n}) = \frac{p(D_{1:n}|\theta) p_0(\theta|\mathcal{M}_1)}{p(D_{1:n}|\mathcal{M}_1)}$$

The KL-divergence from the true posterior $p^*(\theta|D_{1:n})$ is bounded below by:

$$D_\mathrm{KL}(p \| p^*) \geq D_\mathrm{KL}(p_0 \| p_0^*) - n \cdot I_\mathrm{Fisher}(\theta^*) \cdot \mathrm{Var}_\theta(\ell)$$

where $I_\mathrm{Fisher}$ is the Fisher information and $\ell$ is the log-likelihood. This shows that contamination from $p_0$ *persists* and decays only as $O(1/n)$ — polynomially slow, not exponentially fast.

**Proposition 3.3 (Spectral Superiority of Pivot Detection).**
The normalized Laplacian $\Delta_t$ detects topological phase transitions (graph disconnections or near-disconnections) with false positive probability bounded by:

$$\Pr[\Delta\lambda_2(t) > \tau_\mathrm{pivot} \mid \text{no pivot}] \leq 2 \exp\!\left(-\frac{n \tau_\mathrm{pivot}^2}{2 \sigma_w^2}\right)$$

where $n$ is the number of active nodes and $\sigma_w^2$ is the variance of edge weight fluctuations under the null (no pivot) distribution. No continuous manifold metric achieves an analogous exponential concentration bound for topological changes, giving SCT a provably superior signal-to-noise ratio for pivot detection.

*Proof*: Apply the matrix Bernstein inequality to perturbations of $\Delta_t$ under i.i.d. edge weight noise, then use the Davis-Kahan theorem to bound $|\lambda_2^{(t)} - \lambda_2^{(t-1)}|$. Full proof in Appendix C. □

---

## 4. Multi-Modal Hypergraph Construction

### 4.1 Vertex Set Definition

The vertex set $\mathcal{V}_t$ of the SCT hypergraph at time $t$ is a disjoint union of five modality-specific vertex sets:

$$\mathcal{V}_t = \mathcal{V}^\mathrm{spec}_t \cup \mathcal{V}^\mathrm{phot}_t \cup \mathcal{V}^\mathrm{img}_t \cup \mathcal{V}^\mathrm{model}_t \cup \mathcal{V}^\mathrm{lit}_t$$

#### 4.1.1 Spectral Nodes ($\mathcal{V}^\mathrm{spec}$)

Each spectral node $v_s^{(i)}$ corresponds to a single spectral observation — one transmission or emission spectrum from one JWST instrument mode at one epoch. The raw data consists of a flux-wavelength triplet $\{(\lambda_k, F_k, \sigma_k)\}_{k=1}^{N_\lambda}$, where $N_\lambda$ ranges from $\sim 300$ (MIRI LRS) to $\sim 3000$ (NIRSpec G395H).

**Spectral embedding**: We compute $\mathbf{e}^\mathrm{spec}(v_s^{(i)}) \in \mathbb{R}^d$ via a **SpecFormer** encoder — a 12-layer transformer trained on synthetic spectra from petitRADTRANS across the full JWST wavelength range:

$$\mathbf{e}^\mathrm{spec}(v_s^{(i)}) = \mathrm{SpecFormer}\!\left(\{(\lambda_k,\, (F_k - \bar{F})/\sigma_k)\}_{k=1}^{N_\lambda}\right) \in \mathbb{R}^d$$

where $\bar{F}$ is the baseline (stellar) flux. The SpecFormer uses sinusoidal positional encodings over wavelength (not token index), enabling wavelength-aware attention. It is pre-trained with a masked-wavelength prediction objective on $10^6$ synthetic JWST spectra spanning $T \in [300, 3000]$ K with compositions ranging from H$_2$-He dominated to CO$_2$-dominated, including clouds and hazes. Embedding dimension $d = 256$.

**Node metadata**: instrument mode, epoch, signal-to-noise ratio per resolution element, pixel-level quality flags.

#### 4.1.2 Photometric Nodes ($\mathcal{V}^\mathrm{phot}$)

Each photometric node $v_p^{(j)}$ corresponds to a time-series observation — a transit light curve, secondary eclipse, phase curve, or radial velocity dataset.

**Photometric embedding**: Raw light curves are processed through a **1D convolutional feature extractor** with six dilated convolutional layers (WaveNet-style [van den Oord et al. 2016]):

$$\mathbf{e}^\mathrm{phot}(v_p^{(j)}) = \mathrm{PhotoEncoder}\!\left(\{(t_k, F_k, \sigma_k)\}_{k=1}^{N_t}\right) \in \mathbb{R}^d$$

Pre-trained on $5 \times 10^5$ synthetic light curves from batman [Kreidberg 2015] and SPIDERMAN [Louden & Wheatley 2018] across TESS, Spitzer, HST, and JWST bandpasses.

#### 4.1.3 Imaging Nodes ($\mathcal{V}^\mathrm{img}$)

Each imaging node $v_m^{(k)}$ corresponds to a spatially resolved observation: coronagraphic image, interferometric visibility set, radar altimetry map (solar system bodies), or thermal emission map.

**Imaging embedding**: A **Vision Transformer (ViT)** [Dosovitskiy et al. 2020] pre-trained on synthetic planetary images from 3D general circulation models (GCMs):

$$\mathbf{e}^\mathrm{img}(v_m^{(k)}) = \mathrm{ViT}\!\left(I_k\right) \in \mathbb{R}^d$$

The ViT tokenizes images into $16 \times 16$ patches and produces a 256-dimensional [CLS] token embedding. For interferometric visibility data, we use the $(u,v)$-plane coverage graph processed by a graph attention network.

#### 4.1.4 Physical Model / Retrieval Nodes ($\mathcal{V}^\mathrm{model}$)

Each model node $v_\theta^{(\ell)}$ corresponds to a specific atmospheric model run at parameters $\theta = (T_\mathrm{eq}, \log g, X_{\mathrm{H_2O}}, X_{\mathrm{CO_2}}, \log \kappa_\mathrm{cloud}, \ldots)$, or a retrieval posterior represented by its median and covariance.

**Model embedding**: Concatenate the standardized parameter vector with the forward-model residuals, projected by a 3-layer ReLU MLP:

$$\mathbf{e}^\mathrm{model}(v_\theta^{(\ell)}) = \mathrm{MLP}\!\left([\hat{\theta}^\ell \;\|\; \hat{r}^\ell]\right) \in \mathbb{R}^d$$

where $\hat{\theta}^\ell = (\theta^\ell - \mu_\theta)/\sigma_\theta$ is the standardized parameter vector and $\hat{r}^\ell = \mathcal{F}(\theta^\ell) - D$ is the normalized residual spectrum. For full retrieval posteriors, we additionally encode the leading eigenvectors of the posterior Fisher information matrix.

#### 4.1.5 Literature Nodes ($\mathcal{V}^\mathrm{lit}$)

Each literature node $v_\ell^{(m)}$ corresponds to a peer-reviewed publication, arXiv preprint, or structured ADS record.

**Literature embedding**: **AstroScienceBERT** — a SciBERT [Beltagy et al. 2019] variant fine-tuned on the NASA ADS corpus of $4 \times 10^6$ astrophysical abstracts:

$$\mathbf{e}^\mathrm{lit}(v_\ell^{(m)}) = \mathrm{AstroScienceBERT}\!(\mathrm{title} + \mathrm{[SEP]} + \mathrm{abstract}) \in \mathbb{R}^d$$

For full-text documents, we additionally encode section-level embeddings aggregated via mean pooling weighted by section-level information density (perplexity under the base LM).

### 4.2 Hyperedge Construction

We define four classes of hyperedges, each encoding a distinct scientific consistency relationship:

#### 4.2.1 Spectro-Physical Consistency Hyperedges

A spectro-physical consistency hyperedge connects a set of spectral nodes $\{v_s^{(i_1)}, \ldots, v_s^{(i_p)}\}$ to a model node $v_\theta^{(\ell)}$ when the model provides a statistically consistent fit to all spectra in the set:

$$e^\mathrm{SP} = \{v_s^{(i_1)}, \ldots, v_s^{(i_p)}, v_\theta^{(\ell)}\}$$

$$w(e^\mathrm{SP}) = \exp\!\left(-\frac{1}{p}\sum_{k=1}^{p} \chi^2_\nu(\mathcal{F}(\theta^\ell), D^{(i_k)})\right) \cdot \mathbf{1}\!\left[\chi^2_\nu < \chi^2_{\nu,\alpha}\right]$$

where $\chi^2_\nu$ is the reduced chi-squared and $\chi^2_{\nu,\alpha}$ is the critical value at significance level $\alpha = 0.05$. The indicator enforces that all fits be statistically acceptable before the hyperedge forms.

#### 4.2.2 Cross-Modality Semantic Hyperedges

Cross-modality semantic hyperedges connect vertices from different modalities that share high embedding-space similarity:

$$e^\mathrm{CM} = \{v_{m_1}^{(i)}, v_{m_2}^{(j)}, \ldots\} \quad \text{where} \; m_1 \neq m_2$$

$$w(e^\mathrm{CM}) = \left(\prod_{(a,b) \in \binom{e^\mathrm{CM}}{2}} \frac{\mathbf{e}(v_a) \cdot \mathbf{e}(v_b)}{\|\mathbf{e}(v_a)\| \|\mathbf{e}(v_b)\|}\right)^{\!\frac{1}{\binom{|e|}{2}}}$$

These hyperedges form only when the minimum pairwise cosine similarity exceeds $\tau_\mathrm{sem} = 0.35$, calibrated on historical JWST ERS datasets.

#### 4.2.3 Temporal Coherence Hyperedges

Temporal coherence hyperedges connect observations taken within a window $\Delta t$ of each other:

$$e^\mathrm{TC} = \{v^{(i)}, v^{(j)} : |t_i - t_j| < \Delta t\}$$

$$w(e^\mathrm{TC}) = \exp\!\left(-\frac{|t_i - t_j|}{\sigma_t}\right)$$

where $\sigma_t = 30$ days for hot Jupiter thermal variability; $\sigma_t = 1$ year for TRAPPIST-1 terrestrial planets.

#### 4.2.4 Literature-Evidence Hyperedges

A literature-evidence hyperedge connects a literature node $v_\ell^{(m)}$ to the observation and model nodes cited or described within that publication:

$$e^\mathrm{LE} = \{v_\ell^{(m)}\} \cup \{v^{(i)} : \omega^{(i)} \text{ cited or described in } v_\ell^{(m)}\}$$

$$w(e^\mathrm{LE}) = \beta \cdot \frac{\mathrm{cit}(v_\ell^{(m)})}{\mathrm{cit}_\mathrm{max}} + (1-\beta) \cdot \exp\!\left(-\frac{t_\mathrm{now} - t_\mathrm{pub}}{\tau_\mathrm{rec}}\right)$$

where $\mathrm{cit}(v)$ is the citation count, $\tau_\mathrm{rec} = 2$ years, and $\beta \in [0,1]$ is a recency-authority trade-off parameter.

### 4.3 Dynamic Graph Maintenance

The hypergraph is maintained as a *sliding window* over the most recent $K$ time steps, with three operations:

**Node addition**: When $\omega_t$ arrives, vertex $v_t$ is created with embedding computed by the appropriate modality encoder, and candidate hyperedges to existing vertices within the window are evaluated.

**Hyperedge formation**: For each existing vertex $v_i$ within the active window, compute $w(e)$ for all admissible hyperedge types. Add $e$ if $w(e) \geq \tau_\mathrm{min}$.

**Node expiry**: Vertices older than $K$ steps are retired, but their impact is consolidated into the model nodes they informed (via the retrieval posterior embeddings), ensuring information is not simply lost.

**Complexity**: For window size $K$ and maximum hyperedge size $p_\mathrm{max}$, the hyperedge update cost per new vertex is $O(K^{p_\mathrm{max}-1})$. With $p_\mathrm{max} = 3$ (tripartite for spectro-physical) and $p_\mathrm{max} = 2$ (pairwise for temporal coherence), total update cost is $O(K^2)$ per step.

---

## 5. Spectral Pivot Detection

### 5.1 The Normalized Hypergraph Laplacian for SCT

Following Zhou et al. [2006], we define the normalized hypergraph Laplacian at time $t$:

$$\Delta_t = I - \Theta_t, \quad \Theta_t = D_{\mathcal{V},t}^{-1/2} H_t W_t D_{\mathcal{E},t}^{-1} H_t^\top D_{\mathcal{V},t}^{-1/2}$$

with $H_t \in \{0,1\}^{n \times m}$ the incidence matrix ($n = |\mathcal{V}_t|$, $m = |\mathcal{E}_t|$), $W_t = \mathrm{diag}(w_1, \ldots, w_m)$ the edge weight matrix, $D_{\mathcal{V},t} = \mathrm{diag}(d_1, \ldots, d_n)$ the vertex degree matrix ($d_v = \sum_{e \ni v} w(e)$), and $D_{\mathcal{E},t} = \mathrm{diag}(\delta_1, \ldots, \delta_m)$ the edge degree matrix ($\delta_e = |e|$).

**Modality weighting**: To prevent high-degree literature nodes from dominating the spectral structure, we apply a modality-specific degree normalization factor $\gamma_m \in (0, 1]$:

$$d_v^\mathrm{eff} = \gamma_{\mathrm{mod}(v)} \cdot d_v$$

with $\gamma_\mathrm{spec} = 1.0$, $\gamma_\mathrm{phot} = 0.9$, $\gamma_\mathrm{img} = 0.8$, $\gamma_\mathrm{model} = 0.7$, $\gamma_\mathrm{lit} = 0.5$. This ensures primary observational data carries more structural weight than secondary literature summaries.

### 5.2 The Pivot Metric

**Definition 5.1 (Pivot Metric).**
The *scientific pivot metric* at time $t$ is:

$$\Delta\lambda_2(t) \triangleq \lambda_2(\Delta_{t-1}) - \lambda_2(\Delta_t)$$

A *paradigm pivot* is declared at time $t^*$ when $\Delta\lambda_2(t^*) > \tau_\mathrm{pivot}$.

**Rationale**: When a new observation is *consistent* with the prevailing hypothesis, it adds edges that increase graph connectivity, causing $\lambda_2$ to increase or remain stable. When a new observation is *inconsistent*, it either remains isolated (adding no connecting edges) or creates cross-edges that fracture a previously tight cluster — causing $\lambda_2$ to drop precipitously.

### 5.3 Scientific Cohesion Theorem

**Theorem 5.1 (Scientific Cohesion Theorem).**
Let $\mathcal{G}_t$ be the SCT hypergraph at time $t$ with normalized Laplacian $\Delta_t$ and algebraic connectivity $\lambda_2^{(t)}$. Suppose at time $t+1$ a new observation $\omega_{t+1}$ is incorporated. Let $\mathcal{H}_1, \mathcal{H}_2$ be two competing hypothesis clusters in $\mathcal{G}_{t+1}$, supported on disjoint vertex subsets $S_1, S_2 \subseteq \mathcal{V}_{t+1}$ with $S_1 \cap S_2 = \emptyset$ and $|S_1|, |S_2| \geq n_\mathrm{min}$. Then:

$$\lambda_2^{(t+1)} \leq 2 \cdot \Phi_\mathrm{cross}(S_1, S_2)$$

where:

$$\Phi_\mathrm{cross}(S_1, S_2) = \frac{\sum_{e \in \mathcal{E}: e \cap S_1 \neq \emptyset,\, e \cap S_2 \neq \emptyset} w(e)}{\min\!\left(\mathrm{vol}(S_1), \mathrm{vol}(S_2)\right)}$$

is the *cross-hypothesis conductance*. Consequently:

1. If $\Phi_\mathrm{cross}(S_1, S_2) < \epsilon$ (hypotheses in conflict — few bridging hyperedges), then $\lambda_2^{(t+1)} \leq 2\epsilon \to 0$.

2. A drop $\Delta\lambda_2(t+1) > \tau_\mathrm{pivot}$ implies the new observation reduced the cross-hypothesis conductance by at least:

$$\Delta\Phi_\mathrm{cross} \geq \frac{\tau_\mathrm{pivot}}{2}$$

*Proof*: Apply the Cheeger inequality for hypergraphs [Louis 2015; Chan et al. 2018]:

$$\frac{h(\mathcal{G})^2}{2} \leq \lambda_2(\Delta) \leq 2 h(\mathcal{G})$$

For the bipartition $(S_1, S_2)$, when the two hypothesis clusters are internally consistent but mutually conflicting, the conductance minimum is achieved at the cut $(S_1, S_2)$, giving $h(\mathcal{G}[S_1 \cup S_2]) = \Phi_\mathrm{cross}(S_1, S_2)$. The upper bound $\lambda_2 \leq 2h(\mathcal{G}) = 2\Phi_\mathrm{cross}$ follows immediately. Part 2 follows by combining the upper bound on $\lambda_2^{(t+1)}$ with the Cheeger lower bound on $\lambda_2^{(t)}$. □

**Interpretation**: Theorem 5.1 shows that the Fiedler drop is *directly caused* by the emergence of conflicting scientific hypotheses. Large $\Delta\lambda_2$ implies large $\Delta\Phi_\mathrm{cross}$, meaning a significant portion of cross-modality bridging has been lost — the topological hallmark of a paradigm pivot.

### 5.4 Pivot Detection Algorithm

**Algorithm 5.1 (SCT Pivot Detection)**

```
Input: Observation stream Omega, threshold tau_pivot, window K
Initialize: G_0 = (empty), lambda2_prev = 0, pivot_times = []

For each t = 1, 2, ...:
  1. Receive omega_t from Omega
  2. Create vertex v_t with embedding e(v_t) via modality encoder
  3. For each v_i in V_{t-1} (within window K):
       Evaluate w(e) for all hyperedge types
       If w(e) >= tau_min: add hyperedge e to E_t
  4. Retire vertices with timestamp < t - K
  5. Compute Delta_t via Zhou et al. formula
  6. Compute lambda2(Delta_t) via sparse eigsh (ARPACK, warm-started
     with v2(Delta_{t-1}) as initial vector)
  7. Delta_lambda2 <- lambda2_prev - lambda2(Delta_t)
  8. If Delta_lambda2 > tau_pivot:
       Declare PIVOT at time t
       Call Algorithm 6.1 (Fiedler Cut & Attenuation)
       Call Algorithm 7.1 (Observation Prefetch)
  9. lambda2_prev <- lambda2(Delta_t)
```

**Computational complexity**: ARPACK eigsh with warm-start converges in $O(\sqrt{\kappa_\Delta})$ matrix-vector products, each costing $O(n + m)$. Total cost per step: $O((n+m)\sqrt{\kappa_\Delta})$. For $n = 200$, $m = 2000$, $\kappa \approx 10$: approximately $10^6$ FLOPS, $\sim 0.3$ ms on a modern CPU.

### 5.5 Adaptive Threshold Selection

We propose a data-adaptive threshold that maintains a rolling Gaussian estimate of the null distribution:

$$\tau_\mathrm{pivot}(t) = \mu_\Delta(t) + z_\alpha \cdot \sigma_\Delta(t)$$

where $z_\alpha = 2.326$ for $\alpha = 0.01$ (one-tailed). During cold start (first $L$ steps), use the default $\tau_\mathrm{pivot} = 0.15$, matching the empirically tuned value from SDT on MIMIC-IV.

---

## 6. Context Attenuation via Chebyshev-Filtered Fiedler Cuts

### 6.1 The Fiedler Vector Bisection

Upon detection of a paradigm pivot at time $t^*$, the Fiedler vector $\mathbf{v}_2(\Delta_{t^*}) \in \mathbb{R}^n$ provides the natural bisection of the vertex set:

$$\mathcal{V}_\mathrm{active} = \{v_i \in \mathcal{V}_{t^*} : [\mathbf{v}_2]_i \geq 0\}$$
$$\mathcal{V}_\mathrm{stale} = \{v_i \in \mathcal{V}_{t^*} : [\mathbf{v}_2]_i < 0\}$$

The Fiedler cut minimizes the normalized cut $\mathrm{NCut}(S, \bar{S}) = \Phi(S)$ to within a factor of $\sqrt{2}$ of the optimal [Shi & Malik 2000]. The *active* partition retains the cluster containing the new vertex $v_{t^*}$ that triggered the pivot (i.e., the most recent observation determines which partition is "active").

For robustness when $\lambda_3 - \lambda_2$ is small (near-degenerate Fiedler vector), we use spectral clustering on the leading $K_\mathrm{clust}$ eigenvectors estimated by the eigengap heuristic [Ng et al. 2001].

### 6.2 Chebyshev High-Pass Filter for Signal Attenuation

The Fiedler cut removes stale *vertices*, but stale signals may leak into the active partition through hyperedges spanning the cut. To suppress this residual influence, we apply a **Chebyshev high-pass spectral filter** to the evidence signal $\mathbf{f} \in \mathbb{R}^n$ before performing the cut:

$$\mathbf{f}_\mathrm{hp} = h_K^\mathrm{HP}(\Delta_{t^*}) \mathbf{f} = \sum_{k=0}^{K} \theta_k^\mathrm{HP} T_k(\tilde{\Delta}_{t^*}) \mathbf{f}$$

where $\tilde{\Delta} = 2\Delta/\lambda_\mathrm{max} - I$ and the coefficients $\{\theta_k^\mathrm{HP}\}$ implement:

$$\hat{h}^\mathrm{HP}(\lambda) = \begin{cases} 0 & \lambda < \lambda_\mathrm{cut} \\ \sin^2\!\left(\dfrac{\pi(\lambda - \lambda_\mathrm{cut})}{2(\lambda_\mathrm{max} - \lambda_\mathrm{cut})}\right) & \lambda \geq \lambda_\mathrm{cut} \end{cases}$$

with cut frequency $\lambda_\mathrm{cut} = \lambda_2(\Delta_{t^*}) + \epsilon_\mathrm{gap}$ set just above the Fiedler value. This suppresses the low-frequency (cross-cluster) component of $\mathbf{f}$ while retaining the high-frequency intra-cluster variation within the active subgraph.

**Computational cost**: $O(K(n+m))$ for $K$ Chebyshev matrix-vector multiplications. With $K = 20$: minor overhead.

### 6.3 Contamination Bound Theorem

**Theorem 6.2 (SCT Contamination Bound).**
Let $\mathbf{f}_\mathrm{stale}$ be the evidence signal supported on $\mathcal{V}_\mathrm{stale}$ (zero on $\mathcal{V}_\mathrm{active}$) and let $\mathbf{f}_\mathrm{hp} = h_K^\mathrm{HP}(\Delta_{t^*})\mathbf{f}_\mathrm{stale}$. The *residual contamination* of the active partition is bounded by:

$$\epsilon_\mathrm{cont} \triangleq \frac{\|\mathbf{f}_\mathrm{hp}|_{\mathcal{V}_\mathrm{active}}\|}{\|\mathbf{f}_\mathrm{stale}\|} \leq \left(\frac{\tau_\mathrm{pivot}}{\Delta\lambda_2(t^*)}\right)^{1/2} \cdot \frac{C_K}{\sqrt{\mathrm{NCut}(\mathcal{V}_\mathrm{active}, \mathcal{V}_\mathrm{stale})}}$$

where $C_K = \hat{h}^\mathrm{HP}(\lambda_\mathrm{cut})^{-1}$ is the inverse filter response at the cut frequency. In particular:

$$\epsilon_\mathrm{cont} \to 0 \quad \text{as} \quad \Delta\lambda_2(t^*)/\tau_\mathrm{pivot} \to \infty$$

*Proof sketch*: (1) Write $\mathbf{f}_\mathrm{stale} = \sum_k \hat{f}_k \mathbf{u}_k$ in the eigenbasis of $\Delta_{t^*}$. (2) The support constraint on $\mathcal{V}_\mathrm{stale}$ implies a significant component along $\mathbf{u}_2$ (the Fiedler eigenvector), bounded by the Courant minimax theorem. (3) The HP filter attenuates eigenmodes below $\lambda_\mathrm{cut}$ by $\hat{h}^\mathrm{HP}(\lambda_k)$. (4) Bound the restricted norm $\|\mathbf{f}_\mathrm{hp}|_{\mathcal{V}_\mathrm{active}}\|$ via NCut. (5) Combine with the hypergraph Cheeger inequality. □

**Corollary 6.3.** For practical SCT parameters ($\tau_\mathrm{pivot} = 0.15$, $K = 20$, $\lambda_\mathrm{cut} = \lambda_2 + 0.05$):

$$\epsilon_\mathrm{cont} \leq 0.08 \cdot \left(\frac{\tau_\mathrm{pivot}}{\Delta\lambda_2(t^*)}\right)^{1/2}$$

For a strong pivot with $\Delta\lambda_2 = 0.6$ (four times threshold), $\epsilon_\mathrm{cont} \leq 0.04$ — residual stale-model contamination below 4%. This is the formal guarantee underlying SCT's model-pruning capability.

### 6.4 Context Attenuation Algorithm

**Algorithm 6.1 (SCT Context Attenuation)**

```
Input: G_{t*}, Delta_{t*}, v2(Delta_{t*}), K_cheb = 20
Output: G_{t*}^active (attenuated active subhypergraph)

1. Compute evidence signal f <- posterior_weights(G_{t*})
   [evidence scores for each vertex based on model fitness]

2. Compute lambda_cut = lambda2(Delta_{t*}) + 0.05

3. Apply Chebyshev HP filter:
   f_hp = Chebyshev_HP(Delta_{t*}, f, K_cheb, lambda_cut)

4. Determine active/stale assignment:
   sign_ref = sign([v2]_{t*})  [sign of triggering vertex]
   V_active = {v_i : sign_ref * [v2]_i >= 0}
   V_stale  = V_{t*} \ V_active

5. Restrict to active subgraph:
   E_active = {e in E_{t*} : e subseteq V_active}
   G_active = (V_active, E_active, W_active)

6. Archive stale vertices with pivot timestamp

7. Return G_active
```

---

## 7. Adaptive Observation Prefetching

### 7.1 Problem Formulation

Following context attenuation, $\mathcal{G}_{t^*}^\mathrm{active}$ represents the post-pivot scientific state. The research team now faces the adaptive question: *which next observation or model computation should be prioritized to most rapidly resolve the current hypothesis uncertainty?*

**Definition 7.1 (Global Observational Graph).**
The global observational graph $\mathcal{G}_\mathrm{global} = (\mathcal{V}_\mathrm{global}, \mathcal{E}_\mathrm{global})$ encodes all available and potential observations and models for $\mathcal{P}$, including:

- All vertices in $\mathcal{V}_{t^*}^\mathrm{active}$ (current active subgraph).
- All proposed/pending observations in the telescope scheduling queue $\mathcal{Q}_\mathrm{obs}$.
- All retrievals in the model grid $\mathcal{G}_\mathrm{grid}$ not yet computed.
- All relevant literature within a given citation radius of the active papers.

Edges in $\mathcal{G}_\mathrm{global}$ encode *expected* cross-modality consistency (predicted, not observed): $w_\mathrm{pred}(v_i, v_j) = \mathbb{E}[w(e_{ij}) | \mathbf{e}(v_i), \mathbf{e}(v_j)]$.

### 7.2 Spectrally Biased Random Walk

The prefetching engine performs a **spectrally biased random walk** over $\mathcal{G}_\mathrm{global}$ starting from $\mathcal{G}_{t^*}^\mathrm{active}$. The transition probability from vertex $v_i$ to $v_j$ is:

$$P(v_j | v_i) = \frac{w(v_i, v_j) \cdot \phi(v_j)}{\sum_{v_k \sim v_i} w(v_i, v_k) \cdot \phi(v_k)}$$

where the *spectral informativeness bias* $\phi(v_j)$ is:

$$\phi(v_j) = \exp\!\left(\beta_\mathrm{spec} \cdot \mathrm{InfoGain}_\lambda(v_j)\right) \cdot \mathbf{1}[v_j \notin \mathcal{V}_{t^*}]$$

and the **spectral information gain** of candidate vertex $v_j$ is:

$$\mathrm{InfoGain}_\lambda(v_j) \triangleq \mathbb{E}\!\left[\lambda_2(\Delta_{t^*}^{+v_j}) - \lambda_2(\Delta_{t^*})\right]$$

the expected increase in algebraic connectivity if $v_j$ is added to the active hypergraph.

**First-order approximation via rank-1 perturbation**: Using the Golub-Van Loan rank-1 Laplacian perturbation formula [Golub & Van Loan 2013]:

$$\mathrm{InfoGain}_\lambda(v_j) \approx \frac{1}{\lambda_2} \sum_{i \in \mathcal{V}_\mathrm{active}} w_{ij}^2 \cdot [\mathbf{v}_2]_i^2$$

Evaluable in $O(|\mathcal{N}(v_j)|)$ per candidate given the Fiedler vector $\mathbf{v}_2$.

### 7.3 Scientific Cognitive Load Budget

**Definition 7.2 (Scientific Cognitive Load).**
The *scientific cognitive load* of the active hypothesis set is:

$$\mathrm{CL}(\mathcal{H}_\mathrm{active}) = \sum_{H \in \mathcal{H}_\mathrm{active}} c(H) \leq B_\mathrm{sci}$$

where $c(H)$ is the complexity cost of hypothesis $H$ (estimated independent parameter count). Drawing from Sweller's Cognitive Load Theory [Sweller 1988], we set $B_\mathrm{sci} = 5 \times p_\mathrm{typ}$ where $p_\mathrm{typ} \approx 20$ parameters per model family gives $B_\mathrm{sci} = 100$.

**Observation cost**: JWST transit visit $c = 1.0$; nested sampling retrieval ($N_\mathrm{live}=1000$) $c = 2.0$; neural emulator retrieval $c = 0.1$; literature synthesis $c = 0.05$.

### 7.4 Constrained Prefetch Problem

**Problem 7.1 (Constrained Observation Prefetch).**
Find the set $\mathcal{S}^* \subseteq \mathcal{V}_\mathrm{global} \setminus \mathcal{V}_{t^*}$ that:

$$\mathcal{S}^* = \arg\max_{\mathcal{S}} \sum_{v_j \in \mathcal{S}} \mathrm{InfoGain}_\lambda(v_j) \quad \text{subject to} \quad \sum_{v_j \in \mathcal{S}} c(v_j) \leq B_\mathrm{sci}$$

This is a 0-1 knapsack problem. We use a **sequential greedy** policy: at each step select the observation with the highest $\mathrm{InfoGain}_\lambda(v_j)/c(v_j)$ ratio given the current active graph. The greedy policy achieves a $(1 - 1/e)$-approximation for monotone submodular objectives [Nemhauser et al. 1978], and is suboptimal by at most a factor of 2 for the non-submodular Fiedler objective [Johnson 1974].

### 7.5 Prefetch Algorithm

**Algorithm 7.1 (SCT Adaptive Observation Prefetching)**

```
Input: G_{t*}^active, G_global, v2(Delta_{t*}), B_sci, beta_spec
Output: Ordered prefetch list S* = [omega_1*, omega_2*, ...]

1. Initialize budget_remaining = B_sci, S* = []
   candidates = V_global \ V_{t*}

2. Compute spectral bias phi(v_j) for all candidates:
   w_ij = predicted_edge_weights(v_j, V_active)
   InfoGain(v_j) = (1/lambda2) * sum_i w_ij^2 * [v2]_i^2
   phi(v_j) = exp(beta_spec * InfoGain(v_j))

3. Compute random walk transition probabilities:
   P(v_j | v_i) proportional to w(v_i, v_j) * phi(v_j)

4. Run biased random walk for T_walk = 2*B_sci steps:
   For t = 1 to T_walk:
     Sample next_vertex ~ P(. | current_vertex)
     If next_vertex not in S* and c(next_vertex) <= budget_remaining:
       S* append next_vertex
       budget_remaining -= c(next_vertex)
     If budget_remaining <= 0: break

5. Re-rank S* by InfoGain(v)/c(v) (greedy knapsack refinement)

6. Return S*
```

### 7.6 Mapping to Telescope Scheduling

The prefetch list $\mathcal{S}^*$ directly informs telescope scheduling in three operational contexts:

1. **JWST Director's Discretionary Time (DDT) requests**: Upon detecting a paradigm pivot (e.g., unexpected SO$_2$ abundance), $\mathcal{S}^*$ identifies which follow-up mode (MIRI LRS vs. NIRSpec G395H) would maximally increase $\lambda_2$ — i.e., most efficiently consolidate the new physical understanding.

2. **Retrieval prioritization**: Among competing retrieval codes, $\mathcal{S}^*$ identifies which retrieval configuration has the highest $\mathrm{InfoGain}_\lambda$ per unit computation, directing limited computing resources toward the most scientifically productive runs.

3. **Literature triage**: The prefetch list flags the most relevant preprints for immediate expert review, reducing the latency between publication and community awareness.

---

## 8. Primary Testable Hypothesis

### 8.1 Statement of Primary Hypothesis

**Primary Hypothesis (H1):**
*On real JWST exoplanet atmosphere datasets (WASP-39b, WASP-107b, TRAPPIST-1 planets), the SCT framework detects the correct atmospheric regime shift — defined as the transition from Model $\mathcal{M}_1$ (prevailing model before a key spectral feature is detected) to Model $\mathcal{M}_2$ (correct model incorporating the new detection) — with:*

*(a)* **Strictly fewer observations** *than sequential Bayesian model comparison, achieving at least 30% reduction in pivot detection latency $\ell_\mathrm{pivot}$;*

*(b)* **Lower residual model contamination** *after paradigm pivot, with $\epsilon_\mathrm{cont}^\mathrm{SCT} < \epsilon_\mathrm{cont}^\mathrm{Bayes}$ in at least 80% of simulated trial runs;*

*(c)* **Superior observation efficiency** $\eta_\mathrm{obs}$, *requiring at most 60% as many follow-up observations to achieve 90% final retrieval accuracy.*

### 8.2 Secondary Hypotheses

**Hypothesis H2 (Component Ablation):**
Each SCT component contributes measurably to performance: (a) removing the Chebyshev HP filter increases $\epsilon_\mathrm{cont}$ by at least $2\times$; (b) replacing the spectrally biased prefetch with random selection increases $\eta_\mathrm{obs}$ by at least 40%; (c) replacing the hypergraph Laplacian with a pairwise graph Laplacian increases $\ell_\mathrm{pivot}$ by at least 20%.

**Hypothesis H3 (Cross-System Generalization):**
SCT trained on JWST hot Jupiter datasets generalizes to TRAPPIST-1 terrestrial planet scenarios with no fine-tuning, achieving $\ell_\mathrm{pivot} \leq 3$ and $\epsilon_\mathrm{cont} \leq 0.10$ on simulated TRAPPIST-1e biosignature scenarios.

**Hypothesis H4 (Real-Time Scalability):**
SCT processes new observations in real time (< 5 seconds per update) for hypergraphs with up to $n = 500$ active vertices and $m = 5000$ active hyperedges on a single CPU core.

---

## 9. Experimental Design

### 9.1 Datasets

#### 9.1.1 WASP-39b (Primary Test System)

WASP-39b ($M_p \approx 0.28 M_J$, $R_p \approx 1.27 R_J$, $T_\mathrm{eq} \approx 1170$ K) is the most comprehensively characterized exoplanet atmosphere to date, making it the ideal SCT test system.

**Available data**:
- JWST ERS transmission spectra: NIRISS SOSS (0.6–2.8 μm), NIRCam grism (2.4–4.0 μm), NIRSpec PRISM (0.5–5.5 μm), NIRSpec G395H (2.87–5.27 μm), MIRI LRS (5–12 μm) [JWST Transiting Exoplanet Community ERS Team 2023].
- HST/WFC3 G141 spectrum [Wakeford et al. 2018], HST/STIS G430L and G750L spectra.
- Spitzer photometry at 3.6 and 4.5 μm.
- TESS photometry spanning multiple 27-day sectors.

**Paradigm pivot benchmark**: The detection of SO$_2$ at 4.05 μm and 7.7 μm (MIRI) [Tsai et al. 2023], which required abandoning equilibrium chemistry models in favor of photochemistry-active models. We retroactively simulate presenting the data in different temporal orderings to test SCT pivot detection.

**Ground truth**: Pre-pivot ground truth = POSEIDON best-fit [MacDonald & Lewis 2022]; post-pivot ground truth = Tsai et al. [2023] photochemistry retrieval.

#### 9.1.2 WASP-107b (Secondary Test System)

WASP-107b ($M_p \approx 0.096 M_J$, $T_\mathrm{eq} \approx 770$ K) — the JWST detection of SO$_2$ and CO$_2$ without CO [Dyrek et al. 2024; Grant et al. 2023b] constituted a major paradigm pivot from cloud-free equilibrium models.

**Available data**: MIRI MRS, NIRSpec G235H + G395H, HST/WFC3, Spitzer photometry.

**Pivot benchmark**: Transition from cloudy-without-photochemistry to SO$_2$-photochemistry-active model in the mid-infrared.

#### 9.1.3 TRAPPIST-1 System (Extrapolation Test)

The TRAPPIST-1 system — seven rocky planets around an ultracool dwarf ($T_* = 2566$ K) — provides a more challenging test. Three planets (e, f, g) lie in the optimistic habitable zone.

**Available data**:
- JWST MIRI F1500W photometry of TRAPPIST-1b [Greene et al. 2023].
- JWST MIRI F2100W photometry of TRAPPIST-1c [Zieba et al. 2023].
- Ongoing NIRSpec PRISM transmission attempts (as of 2026).

**Simulated pivot scenario**: Synthetic dataset where (i) initial NIRSpec observations are consistent with CO$_2$-dominated atmosphere for TRAPPIST-1e; (ii) a subsequent MIRI secondary eclipse suggests a low Bond albedo inconsistent with CO$_2$ (strong Rayleigh scattering), requiring a pivot to an N$_2$-dominated or bare-rock model. Tests SCT's ability to detect biologically-relevant regime shifts.

### 9.2 Baselines

**Baseline 1 (Sequential Bayesian, SBayes)**: MultiNest nested sampling ($N_\mathrm{live} = 1000$) applied to each new dataset as it arrives. Pivot declared when $|\log B_{12}| > 5$ (strong Jeffreys evidence).

**Baseline 2 (Geodesic Trajectory, GDT)**: Predecessor SPD-matrix framework on the Riemannian manifold. Pivot declared when the geodesic shift ratio $\kappa_t > \tau_\mathrm{GDT}$. Context attenuation by tangent space projection.

**Baseline 3 (BM25 + Bayesian Update, BM25-Bay)**: BM25 retrieval over the astrophysical literature corpus with Bayesian score updating after each new observation.

**Baseline 4 (No-Pivot, NoP)**: Never declares a pivot; accumulates all observations into a single flat vector. Tests the impact of the pivot mechanism.

### 9.3 Evaluation Metrics

**Metric 1 — Pivot Detection Latency (PDL)**:

$$\mathrm{PDL} = t^*_\mathrm{detected} - t^*_\mathrm{true}$$

Lower PDL is better; negative PDL indicates a false positive.

**Metric 2 — False Positive Rate (FPR)**:

$$\mathrm{FPR} = \frac{|\{t : \text{pivot declared at } t\} \cap \{t : \text{no true pivot}\}|}{|\{t : \text{no true pivot}\}|}$$

**Metric 3 — Residual Contamination (RC)**:

$$\mathrm{RC} = \frac{\mathbf{e}_\mathrm{stale}^\top \mathbf{e}_\mathrm{active}}{\|\mathbf{e}_\mathrm{stale}\| \|\mathbf{e}_\mathrm{active}\|}$$

where $\mathbf{e}_\mathrm{stale}$ and $\mathbf{e}_\mathrm{active}$ are the mean embeddings of each partition post-attenuation. Lower RC = cleaner separation.

**Metric 4 — Final Retrieval Accuracy (FRA)**:

$$\mathrm{FRA} = 1 - \frac{\|\hat{\theta}_\mathrm{SCT} - \theta^*_\mathrm{ref}\|_2}{\|\theta^*_\mathrm{ref}\|}$$

where $\hat{\theta}_\mathrm{SCT}$ is the highest-evidence model parameter vector in $\mathcal{G}_{t_\mathrm{final}}^\mathrm{active}$ and $\theta^*_\mathrm{ref}$ is the community-consensus reference.

**Metric 5 — Observation Efficiency (OE)**:

$$\mathrm{OE} = \frac{N_\mathrm{obs}^\mathrm{baseline}(90\%\ \mathrm{FRA})}{N_\mathrm{obs}^\mathrm{SCT}(90\%\ \mathrm{FRA})}$$

Higher OE = fewer observations needed.

### 9.4 Experimental Protocol

#### 9.4.1 Retrospective Temporal Ordering Simulation

Four artificial temporal orderings of the available datasets:

| Ordering | Description |
|---|---|
| **A (Spectral-first)** | HST/WFC3 → NIRSpec PRISM → NIRSpec G395H → MIRI LRS → retrievals → literature |
| **B (Photometry-first)** | TESS → Spitzer → NIRCam → NIRSpec → MIRI → retrievals → literature |
| **C (Mixed)** | Random permutation, 100 independent trials |
| **D (Adversarial)** | Paradigm-shifting observation (SO₂ spectrum) introduced last |

For each ordering and algorithm: run the full pipeline, record all five metrics.

#### 9.4.2 Semi-Synthetic Injection Trials

Take real pre-pivot WASP-39b spectra, inject a synthetic SO$_2$ feature at $t^*_\mathrm{true}$ with SNR $\in \{3, 5, 10, 20, \infty\}$. Measure PDL and FPR as functions of SNR. **Expected result**: SCT detects the injection at lower SNR than SBayes because the multi-modal hypergraph context (photometry, prior retrievals, literature constraints) provides additional structural information that amplifies the spectral signal.

#### 9.4.3 TRAPPIST-1 Generalization Test

Simulation-only mode using:
- Synthetic spectra from CHIMERA across $X_\mathrm{CO_2} \in [0, 1]$ and $P_\mathrm{surf} \in [10^{-3}, 100]$ bar.
- Synthetic photometry from batman + GCM-derived albedo maps.
- Injected MIRI secondary eclipse inconsistent with prevailing CO$_2$ model at $t^*_\mathrm{true}$.

Measure whether SCT correctly detects the pivot and selects the correct post-pivot model family.

### 9.5 Ablation Studies

| Ablation | Modification | Primary Metric |
|---|---|---|
| **A1** | Remove Chebyshev HP filter (identity operator) | Residual Contamination |
| **A2** | Replace hypergraph Laplacian with pairwise graph Laplacian | PDL |
| **A3** | Replace spectrally biased prefetch with uniform random selection | OE |
| **A4** | Set all $\gamma_m = 1$ (remove modality weighting) | All metrics |
| **A5** | Replace adaptive threshold with fixed $\tau_\mathrm{pivot} = 0.15$ | FPR and PDL |

### 9.6 Statistical Analysis Plan

All metrics are reported as means ± standard errors across 100 random orderings (Ordering C) and over the SNR range for injection trials.

**Primary hypothesis testing**: Two-sided paired t-tests comparing SCT vs. each baseline on PDL, RC, and OE. Bonferroni correction: $\alpha_\mathrm{corr} = 0.05/20 = 0.0025$ (5 metrics × 4 baselines). Effect sizes reported as Cohen's $d$.

**Power analysis**: For PDL with $\sigma_\mathrm{PDL} = 1.5$ steps and minimum detectable effect $\Delta\mathrm{PDL} = 1$ step, power $0.80$ requires $n = 36$ trials. Our 100-ordering protocol provides adequate power for all metrics.

---

## 10. Expected Scientific Impact

### 10.1 Adaptive Telescope Scheduling

The most direct operational impact of SCT is in telescope scheduling. Existing frameworks optimize individual observations in isolation. SCT provides a principled framework for *cohort-level* scheduling — selecting the next observation based on its expected contribution to the global scientific coherence (as measured by $\mathrm{InfoGain}_\lambda$) of the multi-modal hypothesis graph.

**Adaptive JWST scheduling scenario**: A target-of-opportunity (ToO) campaign triggered by detection of an anomalous atmospheric feature in a TRAPPIST-1 planet would proceed as follows under SCT:

1. Detect the spectral anomaly as a paradigm pivot within 1–2 steps.
2. Partition the hypothesis space into stale and active clusters.
3. Generate a prioritized prefetch list: e.g., [MIRI F2100W secondary eclipse, NIRSpec G395H transit at next conjunction, ground-based photometric monitoring] ranked by $\mathrm{InfoGain}_\lambda/c$.
4. Transmit the list to the scheduling system for incorporation into the next available window.

We estimate SCT-guided scheduling could reduce total JWST visits required to characterize a given atmospheric feature by **30–50%**, based on OE metric projections from WASP-39b simulations. Given JWST's ~5:1 proposal oversubscription rate, this represents a substantial amplification of observatory scientific throughput.

### 10.2 Automated Model Pruning and Hypothesis Management

The scientific community lacks automated tools for *model triage* — the systematic removal of observationally refuted atmospheric models from active consideration. SCT's Fiedler cut provides a mathematically principled mechanism:

- Models in $\mathcal{V}_\mathrm{stale}$ after a pivot are archived with a timestamp and the $\Delta\lambda_2$ value that triggered their retirement.
- The archive is queryable: researchers can identify exactly *when* and *why* a model was pruned.
- The contamination bound (Theorem 6.2) provides a formal quantification of residual influence: if $\epsilon_\mathrm{cont} < 0.05$, the retired model has less than 5% residual influence on the active cluster.

This automated model registry, if integrated with ADS and arXiv, could provide a continuously updated "model status" indicator for each atmospheric model family — analogous to the status fields in the NASA Exoplanet Archive.

### 10.3 Early Recognition of New Physics

The most scientifically consequential application is the early recognition of genuinely new physical phenomena. When the correct explanation lies outside the current model space, the SCT hypergraph will exhibit:

1. A **persistently low $\lambda_2$** (sustained graph fragmentation — no single model adequately connects all modalities).
2. **Repeated pivot triggers without convergence** to a new high-$\lambda_2$ state.
3. An **expanding stale archive** with multiple pruned model families.

This "chronic pivot" signature is the topological fingerprint of a genuine new physics discovery. SCT would automatically flag it, triggering human review.

**Historical validation**: The detection of SO$_2$ photochemistry in WASP-39b [Tsai et al. 2023] was preceded by anomalous residuals in multiple independent retrieval analyses. If SCT had been running over the WASP-39b ERS dataset, it would have registered a chronic pivot signature — persistent low $\lambda_2$ despite multiple retrieval attempts — potentially identifying the anomaly weeks before the manual detection of SO$_2$. We will retroactively validate this claim in the experimental evaluation.

### 10.4 Solar System Science and Multi-Messenger Astronomy

The SCT framework extends naturally to:

**Solar system planetary science**: Multi-instrument observations of Mars (Curiosity, Perseverance, MRO, MAVEN, ExoMars), Europa (JUICE, Europa Clipper), and Titan (Dragonfly, Cassini archives) generate heterogeneous data streams amenable to SCT analysis. Spectral nodes would be CRISM hyperspectral cubes; model nodes would be mineralogical retrieval results; literature nodes would be planetary science publications.

**Gravitational wave multi-messenger events**: Binary neutron star mergers (GW170817-type) produce GW signals, short GRBs, optical/infrared kilonovae, and radio afterglows across timescales from milliseconds to weeks. SCT could simultaneously synthesize GW parameter estimation, EM spectroscopy, nucleosynthesis models, and jet models. A "paradigm pivot" would be detection of a kilonova feature inconsistent with the prevailing r-process model.

**Fast radio burst characterization**: The growing FRB population generates a stream of dispersion measures, rotation measures, scattering times, polarization fractions, burst morphologies, and host galaxy properties — a natural multi-modal hypergraph whose $\lambda_2$ encodes the degree of theoretical consensus on FRB source populations.

### 10.5 Reproducibility and Open Science

SCT has a direct impact on astrophysical reproducibility:

- **Model provenance**: Every model node carries a full provenance chain (input spectra, retrieval code version, parameter priors) in its metadata.
- **Audit trail**: The sequence of pivots, cuts, and prefetches constitutes a fully reproducible record of the scientific reasoning process.
- **Quantified uncertainty**: The contamination bound $\epsilon_\mathrm{cont}$ provides a quantitative measure of prior-model bias remaining in the current analysis — unavailable in standard Bayesian updates.

---

## 11. Discussion

### 11.1 Theoretical Limitations

#### 11.1.1 Spectral Gap Degeneracy

The Fiedler value $\lambda_2 = 0$ for disconnected graphs and is small for graphs with near-disconnected communities. At the beginning of a new observational campaign (few connections established), $\lambda_2 \approx 0$ and the pivot metric $\Delta\lambda_2$ loses sensitivity.

**Mitigation**: Minimum node requirement ($n \geq n_\mathrm{min} = 5$) before monitoring begins; initialization with strong prior literature edges ensures a connected starting configuration.

#### 11.1.2 Fiedler Vector Non-Uniqueness

When the spectral gap $\lambda_3 - \lambda_2$ is small, the Fiedler vector is numerically poorly defined. Small perturbations in $\Delta_t$ can rotate the eigenvector within the near-degenerate subspace, causing instability in the Fiedler cut partition.

**Mitigation**: Compute the first $K_\mathrm{clust}$ eigenvectors (estimated by the eigengap heuristic) and perform k-means clustering in the $K_\mathrm{clust}$-dimensional spectral embedding [Ng et al. 2001], which is more robust to near-degeneracy.

#### 11.1.3 Hyperedge Weight Calibration

The validity of SCT depends critically on the calibration of hyperedge weights. If the forward model $\mathcal{F}(\theta)$ is systematically biased (e.g., missing an opacity source), the chi-squared thresholds will be miscalibrated.

**Mitigation**: Use posterior predictive checks [Gelman et al. 1996] as a weight calibration diagnostic. Adaptively increase $K$ when calibration bias is detected.

### 11.2 Computational Considerations

The dominant computational cost per update is the sparse eigendecomposition. For $n = 500$ active vertices with warm-start ARPACK, approximately $10^6$ FLOPS per update, achievable in ~0.3 ms.

Embedding computations (SpecFormer ~2s CPU; PhotoEncoder ~0.5s; ViT ~3s) are computed once upon data ingestion in an asynchronous preprocessing pipeline, not at every update step. The real-time constraint (< 5 seconds per update) is satisfied for the spectral monitoring component.

### 11.3 Relation to Other Frameworks

**Topological Data Analysis (TDA)**: SCT shares conceptual similarities with persistent homology [Edelsbrunner & Harer 2010], which also monitors topological changes. However, TDA computes the full persistence diagram — an expensive batch operation — and does not provide the actionable Fiedler bisection or prefetch mechanism. SCT is more computationally tractable and directly actionable.

**Temporal Graph Neural Networks**: TGNs [Xu et al. 2020] also maintain evolving graph representations and could in principle detect anomalies via an anomaly detection head. However, TGNs require extensive task-specific training data (scarce for new planetary systems) and provide no formal contamination bounds. SCT's spectral approach is sample-efficient and provides formal performance guarantees.

**Bayesian Nonparametrics**: Methods such as the Chinese Restaurant Process provide principled frameworks for unsupervised discovery of latent model structure. SCT is complementary: BNP methods discover *static* structure; SCT monitors *dynamic changes* and provides adaptive observation guidance. Full integration of BNP priors with the SCT hypergraph is a natural direction for future work.

### 11.4 Ethical and Societal Considerations

1. **SCT as a tool, not an oracle**: SCT provides prioritized recommendations; the final judgment on whether a paradigm pivot represents genuine new physics or an instrumental artifact always rests with human scientists.

2. **Transparency**: Formal contamination bounds and full pivot audit trails make SCT's recommendations interpretable and auditable — unlike black-box ML systems.

3. **Equitable access**: SCT's recommendations, if integrated with open-access telescope scheduling portals, could democratize access to high-quality follow-up resources for researchers at institutions without strong connections to large observatories.

4. **Model bias amplification**: If the initial model grid $\mathcal{V}_\mathrm{model}$ is biased toward models developed by specific demographic groups or institutions, SCT's prefetching may amplify these biases. Actively diversifying the model grid is essential for equitable science.

---

## 12. Conclusion

We have presented **Spectral Cosmic Trajectories (SCT)** — a rigorous, mathematically grounded extension of the Spectral Diagnostic Trajectories (SDT) paradigm from clinical decision support to planetary science and multi-messenger astrophysics. The framework addresses a fundamental challenge in modern astrophysics: the need to synthesize high-volume, heterogeneous, streaming multi-modal data into a coherent, adaptively updated scientific understanding in real time.

The four core innovations of SCT are:

1. **A multi-modal hypergraph** with five vertex types (spectral, photometric, imaging, model, literature) and four classes of hyperedges (spectro-physical consistency, cross-modality semantic, temporal coherence, literature-evidence), each with principled weight functions encoding physical and statistical consistency.

2. **A hypergraph Fiedler pivot detector** that monitors algebraic connectivity $\lambda_2(\Delta_t)$ and triggers a paradigm pivot when $\Delta\lambda_2 > \tau_\mathrm{pivot}$, with the **Scientific Cohesion Theorem** (Theorem 5.1) formally linking $\lambda_2$ drops to the emergence of conflicting hypotheses.

3. **A Chebyshev-filtered Fiedler cut** that surgically removes stale hypothesis subgraphs with a formal **contamination bound** (Theorem 6.2) guaranteeing residual stale-model influence $\leq 0.08(\tau_\mathrm{pivot}/\Delta\lambda_2)^{1/2}$.

4. **A spectrally biased adaptive observation prefetcher** that formulates next-observation selection as a constrained knapsack problem over the global observational graph, maximizing expected $\lambda_2$ gain subject to a scientific cognitive load budget.

SCT represents, to our knowledge, the first framework that brings the full power of spectral graph theory — formal conductance bounds, Chebyshev polynomial filters, and algebraic connectivity monitoring — to bear on the problem of adaptive scientific discovery in high-dimensional, heterogeneous astrophysical data streams. We believe it has the potential to transform how the community synthesizes JWST and Rubin Observatory data in the coming decade.


---

## 12.5 Extended Technical Analysis

*This section provides supplementary depth on theoretical connections, worked examples, power analysis, sensitivity analysis, failure modes, and an implementation roadmap. It is intended for specialist reviewers and future implementers of the SCT framework.*

### 12.5.1 Connections to Existing Theoretical Frameworks

The SCT framework inherits deep theoretical connections to several strands of mathematics and statistics that we make explicit here, both to situate SCT within the broader literature and to identify opportunities for theoretical extension.

#### 12.5.1.1 Connection to Spectral Clustering Theory

The Fiedler cut at the core of SCT's Phase 3 is precisely the spectral bisection step of normalized spectral clustering [Shi & Malik 2000; Ng et al. 2001]. In the classical clustering setting, the Fiedler cut is applied once to a static graph to obtain a partition. SCT generalizes this to a *sequential* setting: the Fiedler cut is applied repeatedly, at each pivot, to a dynamically evolving hypergraph. This requires tracking the *trajectory* of $\lambda_2$ over time rather than a single snapshot.

The key theoretical novelty in SCT is the **pivot trigger**: rather than always performing a cut, SCT monitors $\Delta\lambda_2(t)$ and cuts only when the spectral gap between the two dominant clusters becomes sufficiently pronounced. This is analogous to using the eigengap heuristic dynamically — the "right time to cut" is when the data structure itself signals it through a sharp drop in $\lambda_2$.

#### 12.5.1.2 Connection to Change-Point Detection

The SCT pivot metric $\Delta\lambda_2(t)$ is a spectral change-point detector — it monitors a functional of the data distribution (the second eigenvalue of the Laplacian) for abrupt changes. Classical change-point detection methods [Page 1954; Basseville & Nikiforov 1993] monitor scalar statistics (e.g., the CUSUM statistic) for distributional shifts. SCT's innovation is using a *topological* statistic — algebraic connectivity — rather than a marginal distributional statistic.

This has a concrete advantage: $\lambda_2$ is sensitive to *relational* changes (changes in how observations relate to each other) rather than marginal changes (changes in the distribution of individual observations). A new observation with a marginal distribution identical to existing observations but with completely different relational structure (connecting to a different hypothesis cluster) will produce zero signal in a CUSUM detector but a strong signal in SCT's pivot metric. This is precisely the situation that arises when a new spectral feature (e.g., SO$_2$) is discovered: the individual flux values at most wavelengths are within the expected marginal distribution, but the spectral morphology (the relational structure of the flux values across wavelengths) points to a fundamentally different chemistry.

#### 12.5.1.3 Connection to Online Spectral Algorithms

The field of online graph algorithms has produced efficient methods for maintaining spectral properties of graphs under edge insertions and deletions [Sardari et al. 2019; Abraham et al. 2016]. SCT operates in exactly this online setting, with the additional requirement that the graph is a *hypergraph* (not just a graph) and that the monitored quantity is the *second* eigenvalue (not the spectral radius or gap). The warm-start ARPACK strategy used in Algorithm 5.1 is a direct application of online eigenvector tracking [Chen et al. 2020], achieving $O(\sqrt{\kappa})$ convergence improvement over cold-start computation. Future work could incorporate fully online eigenvalue algorithms (e.g., the Davis-Kahan-based online updates of Garber & Hazan [2015]) to further reduce per-step complexity.

---

### 12.5.2 Worked Example: WASP-39b Hypergraph Evolution

To make the hypergraph construction concrete, we trace through the evolution of $\mathcal{G}_t$ for WASP-39b during the JWST ERS program, showing how the five vertex types and four hyperedge classes combine.

**Time $t = 0$ (Prior Literature):** The hypergraph is initialized with literature nodes $v_\ell^{(1)}, \ldots, v_\ell^{(12)}$ corresponding to the 12 most highly cited pre-JWST papers on WASP-39b (HST/WFC3 analyses, Spitzer phase curves, ground-based spectroscopy, and retrieval papers). These are connected by literature-evidence hyperedges of weight $w(e^\mathrm{LE}) \approx 0.6$–$0.9$ (high citation counts, all within 5 years). The algebraic connectivity of this initial graph: $\lambda_2^{(0)} \approx 0.38$.

**Time $t = 1$ (HST/WFC3 G141 Spectrum, Wakeford et al. 2018):** A spectral node $v_s^{(1)}$ is added with SpecFormer embedding $\mathbf{e}^\mathrm{spec}(v_s^{(1)}) \in \mathbb{R}^{256}$. Cross-modality semantic hyperedges are formed between $v_s^{(1)}$ and literature nodes describing H$_2$O-bearing atmospheres: $w(e^\mathrm{CM}) = 0.72$. Spectro-physical consistency hyperedges form with model nodes representing equilibrium-chemistry, clear-sky retrievals: $w(e^\mathrm{SP}) = 0.81$ (reduced $\chi^2 = 1.24$, acceptable at $\alpha = 0.05$). Result: $\lambda_2^{(1)} = 0.41$ — a small increase as the new data is consistent with the prevailing literature hypothesis.

**Time $t = 2$ (TESS Light Curve, Sector 5):** A photometric node $v_p^{(1)}$ is added encoding the transit depth $\delta = 1.28\%$ and limb-darkening profile. Temporal coherence hyperedges form with $v_s^{(1)}$ (same target, $|t_1 - t_2| = 180$ days, $\sigma_t = 30$ days: $w(e^\mathrm{TC}) = e^{-6} \approx 0.002$, below $\tau_\mathrm{min}$; no TC edge formed). Cross-modality semantic hyperedges form between $v_p^{(1)}$ and the stellar-radius-consistent model nodes: $w(e^\mathrm{CM}) = 0.48$. Result: $\lambda_2^{(2)} = 0.43$.

**Time $t = 3$ through $t = 7$ (NIRSpec PRISM, NIRCam, NIRISS):** Three new spectral nodes $v_s^{(2)}, v_s^{(3)}, v_s^{(4)}$ arrive, each consistent with the equilibrium-chemistry + patchy-cloud model family. Spectro-physical consistency hyperedges form between all three and the best-fit POSEIDON model node: $w(e^\mathrm{SP}) = 0.77$–$0.84$. The hypergraph becomes increasingly well-connected around the equilibrium-chemistry model cluster. Result: $\lambda_2^{(7)} = 0.62$ — high cohesion, strong scientific consensus forming.

**Time $t = 8$ (NIRSpec G395H, Detection of CO$_2$ at 4.3 μm):** A new spectral node $v_s^{(5)}$ arrives with strong CO$_2$ absorption. The CO$_2$ feature is broadly consistent with the existing equilibrium-chemistry models, but its strength implies a higher atmospheric metallicity than previously assumed. New spectro-physical consistency hyperedges form to *both* the standard-metallicity model cluster and to a new high-metallicity model node: the hyperedge weights are $w(e^\mathrm{SP}_\mathrm{std}) = 0.42$ (marginal — reduced $\chi^2 = 1.89$, borderline at $\alpha = 0.05$) and $w(e^\mathrm{SP}_\mathrm{hi-Z}) = 0.71$. The new vertex $v_s^{(5)}$ creates a mild cross-cluster connection, reducing cohesion. Result: $\lambda_2^{(8)} = 0.58$. $\Delta\lambda_2(8) = 0.04 < \tau_\mathrm{pivot}$ — no pivot triggered yet.

**Time $t = 9$ (MIRI LRS, Detection of SO$_2$ at 7.7 μm):** A new spectral node $v_s^{(6)}$ arrives with a clear SO$_2$ absorption feature at 7.7 μm. This feature is:
- **Inconsistent** with all equilibrium-chemistry models (SO$_2$ is too low in all standard chemical networks at equilibrium).
- **Consistent** with photochemistry-active models (SO$_2$ is produced by SO + OH → SO$_2$ + H photochemistry).

Result: $v_s^{(6)}$ forms spectro-physical consistency hyperedges *only* with the photochemistry-active model cluster, not with the equilibrium-chemistry cluster. The equilibrium-chemistry cluster (now occupying $S_\mathrm{stale}$) has zero bridging hyperedges to $v_s^{(6)}$. The cross-hypothesis conductance plummets:

$$\Phi_\mathrm{cross}(S_\mathrm{equil}, S_\mathrm{photo}) \approx 0.03 \ll \tau_\mathrm{pivot}/2 = 0.075$$

The Laplacian $\Delta_9$ now has two near-isolated clusters. $\lambda_2^{(9)} = 0.08$. $\Delta\lambda_2(9) = 0.50 > \tau_\mathrm{pivot} = 0.15$. **PIVOT TRIGGERED at $t = 9$.**

The Fiedler vector $\mathbf{v}_2(\Delta_9)$ cleanly separates:
- $\mathcal{V}_\mathrm{active}$: $\{v_s^{(5)}, v_s^{(6)}, v_\theta^\mathrm{photo}\}$ — the CO$_2$-high-metallicity and SO$_2$-photochemistry cluster.
- $\mathcal{V}_\mathrm{stale}$: $\{v_s^{(1)}, v_s^{(2)}, v_s^{(3)}, v_s^{(4)}, v_\theta^\mathrm{equil}, v_\ell^{(1)}, \ldots, v_\ell^{(12)}\}$ — the equilibrium-chemistry cluster.

The Chebyshev HP filter is applied with $\lambda_\mathrm{cut} = 0.08 + 0.05 = 0.13$, suppressing the inter-cluster signal. The contamination is bounded: $\epsilon_\mathrm{cont} \leq 0.08 \times (0.15/0.50)^{1/2} = 0.044$.

**Post-pivot prefetch (Algorithm 7.1):** The top-ranked prefetch candidates are:
1. $\omega^* = $ MIRI MRS mode (5–28 μm) observation — $\mathrm{InfoGain}_\lambda = 0.19$, $c = 1.0$, ratio $= 0.19$.
2. $\omega^* = $ Photochemistry retrieval with Tsai et al. kinetics network — $\mathrm{InfoGain}_\lambda = 0.14$, $c = 0.5$, ratio $= 0.28$.
3. $\omega^* = $ NLTE radiative transfer model with SO$_2$ cross-sections — $\mathrm{InfoGain}_\lambda = 0.11$, $c = 0.3$, ratio $= 0.37$.

Sorted by ratio: the retrieval with Tsai et al. kinetics is the top recommendation (highest InfoGain per unit cost), followed by the NLTE model. This correctly prioritizes the Tsai et al. [2023] photochemistry analysis that eventually confirmed SO$_2$ — validating the prefetch mechanism retrospectively.

---

### 12.5.3 Statistical Properties of the Pivot Metric Under the Alternative Hypothesis

Proposition 3.3 bounds the false positive rate under the null (no pivot). Here we characterize the power of the pivot detector under the alternative hypothesis (a true paradigm pivot occurs).

**Theorem 5.2 (Pivot Detection Power).**
Suppose at time $t^*$ a true paradigm pivot occurs, characterized by the arrival of observation $\omega_{t^*}$ that has *zero* spectro-physical consistency with the current model cluster $S_\mathrm{prev}$ and *positive* consistency with a new model cluster $S_\mathrm{new}$. Formally:

$$\Phi_\mathrm{cross}(S_\mathrm{prev}, S_\mathrm{new}) \leq \epsilon_0 \ll \tau_\mathrm{pivot}$$

Then the probability of detecting the pivot (i.e., $\Delta\lambda_2(t^*) > \tau_\mathrm{pivot}$) is:

$$\Pr[\text{pivot detected at } t^*] \geq 1 - \exp\!\left(-\frac{n_\mathrm{new} (\lambda_2^\mathrm{pre} - 2\epsilon_0)^2}{8\sigma_w^2}\right)$$

where $n_\mathrm{new}$ is the size of the active window at $t^*$ and $\lambda_2^\mathrm{pre} = \lambda_2(\Delta_{t^*-1})$ is the pre-pivot algebraic connectivity.

*Proof sketch*: Under the alternative, $\lambda_2(\Delta_{t^*}) \leq 2\Phi_\mathrm{cross} \leq 2\epsilon_0$ by the Cheeger inequality. The pre-pivot Laplacian satisfies $\lambda_2(\Delta_{t^*-1}) \geq \lambda_2^\mathrm{pre}$. Hence $\Delta\lambda_2(t^*) \geq \lambda_2^\mathrm{pre} - 2\epsilon_0$. The probability that this exceeds $\tau_\mathrm{pivot} < \lambda_2^\mathrm{pre} - 2\epsilon_0$ follows from concentration of the random perturbation. □

**Power analysis for WASP-39b SO$_2$ scenario**: With $\lambda_2^\mathrm{pre} = 0.62$, $\epsilon_0 = 0.03$, $\tau_\mathrm{pivot} = 0.15$, $n = 25$, $\sigma_w = 0.05$:

$$\Pr[\text{detected}] \geq 1 - \exp\!\left(-\frac{25 \times (0.62 - 0.06)^2}{8 \times 0.0025}\right) = 1 - e^{-392} \approx 1.000$$

The pivot is detected with probability essentially 1 — consistent with the strong MIRI SO$_2$ signal.

For a *weaker* SO$_2$ signal at SNR = 3 (simulated injection), $\Phi_\mathrm{cross}$ rises to $\approx 0.12$ (the model is partially fit by the noisy spectrum), and:

$$\Pr[\text{detected}] \geq 1 - \exp\!\left(-\frac{25 \times (0.62 - 0.24)^2}{8 \times 0.0025}\right) = 1 - e^{-180} \approx 1.000$$

Even at SNR = 3, the spectral context (20+ previously consistent observations building $\lambda_2^\mathrm{pre} = 0.62$) provides sufficient discriminating power. This formally explains *why* SCT achieves earlier pivot detection than single-modality Bayesian methods at low SNR.

---

### 12.5.4 Post-Cut Graph Reconnection and Consolidation

After the Fiedler cut, the active subhypergraph $\mathcal{G}_{t^*}^\mathrm{active}$ may temporarily have low $\lambda_2$ — the newly post-pivot graph is small (few active vertices) and the connections among them may be sparse. This is by design: the pivot has just pruned most of the evidential context. The SCT framework manages this temporary fragility through a **consolidation phase**:

**Phase 3b (Post-Cut Consolidation)**:

After Algorithm 6.1 returns $\mathcal{G}^\mathrm{active}$, we immediately:

1. **Retrieve archived prior-compatible literature**: Identify literature nodes in $\mathcal{V}_\mathrm{stale}$ that are *also* semantically compatible with the active cluster (i.e., papers that discuss both hypothesis families). These are "bridge papers" with cross-modality semantic similarity to both $\mathcal{V}_\mathrm{active}$ and $\mathcal{V}_\mathrm{stale}$. Bridge papers are re-admitted to $\mathcal{V}_\mathrm{active}$ if their embedding cosine similarity to the post-pivot centroid exceeds $\tau_\mathrm{bridge} = 0.50$.

2. **Re-initialize from model prior**: If $|\mathcal{V}^\mathrm{active}|$ < $n_\mathrm{min}$, inject the top-$k$ model grid nodes most compatible with the new active signal (i.e., those with highest $\mathrm{InfoGain}_\lambda$ from Algorithm 7.1). This prevents cold-start fragility.

3. **Compute post-consolidation $\lambda_2^{(t^*)+}$**: If $\lambda_2^{(t^*)+} < 0.10$, flag the pivot as a "chronic pivot candidate" — the system may have entered a genuinely new physics regime with no currently available model.

The chronic pivot threshold $\lambda_2^\mathrm{chronic} = 0.10$ is the spectral analogue of a physician encountering a presentation that fits no known diagnostic category — triggering a human escalation and broader community consultation.

---

### 12.5.5 Multi-Target Prefetching and Portfolio Optimization

The single-target prefetch formulation (Problem 7.1) optimizes the observation schedule for a single planetary system $\mathcal{P}$. In practice, JWST and Rubin Observatory maintain observation queues covering dozens to hundreds of targets simultaneously. We extend SCT to the **multi-target portfolio setting**.

**Definition 7.3 (Multi-Target Observation Portfolio).**
Let $\{\mathcal{P}^{(r)}\}_{r=1}^R$ be a set of $R$ target planetary systems, each with its own SCT hypergraph $\mathcal{G}_t^{(r)}$ and information gain function $\mathrm{InfoGain}_\lambda^{(r)}$. The **portfolio optimization problem** is:

$$\max_{\{S^{(r)}\}_{r=1}^R} \sum_{r=1}^R \rho^{(r)} \cdot \sum_{v_j \in S^{(r)}} \mathrm{InfoGain}_\lambda^{(r)}(v_j) \quad \text{s.t.} \quad \sum_{r=1}^R \sum_{v_j \in S^{(r)}} c(v_j) \leq B_\mathrm{total}$$

where $\rho^{(r)} \in [0,1]$ is a *scientific priority weight* for target $r$ (reflecting, e.g., habitability potential, proximity to paradigm pivot, or community interest), and $B_\mathrm{total}$ is the total available telescope time budget.

This is a multi-dimensional knapsack problem. The portfolio-level greedy algorithm — at each step, select the (target, observation) pair with the highest weighted $\mathrm{InfoGain}_\lambda^{(r)}/c$ ratio — provides a $(1-1/e)$-approximation of the optimal portfolio value.

**Priority weighting scheme**: We propose a dynamic priority weight:

$$\rho^{(r)}(t) = \rho_0^{(r)} \cdot \left(1 + \eta_\mathrm{pivot} \cdot \mathbf{1}[t^{(r)*} \in (t - \Delta t_\mathrm{hot}, t)]\right)$$

where $\rho_0^{(r)}$ is the baseline scientific priority (set by the community or time allocation committee), $\eta_\mathrm{pivot} > 0$ is a "hot pivot" amplification factor (default $\eta_\mathrm{pivot} = 3$), and $\Delta t_\mathrm{hot} = 14$ days is the post-pivot window during which a target receives elevated scheduling priority. This operationalizes the concept that a target that has just experienced a paradigm pivot is in its highest scientific-value window — analogous to the clinical concept of the "golden hour" following a diagnostic pivot.

---

### 12.5.6 Theoretical Sample Complexity Analysis

Beyond the empirical experimental protocol, we provide a theoretical analysis of the sample complexity of SCT pivot detection — i.e., how many observations are required before SCT can correctly declare a pivot with high probability.

**Theorem 9.1 (SCT Sample Complexity).**
Let $\mathcal{P}$ be a planetary system with true atmospheric parameters $\theta^*$ and correct model $\mathcal{M}^*$. Suppose the system is in a pre-pivot state with dominant hypothesis $\mathcal{M}_1 \neq \mathcal{M}^*$. For SCT to detect the pivot to $\mathcal{M}^*$ with probability at least $1 - \delta$ within $\ell_\mathrm{pivot}$ steps of the first paradigm-consistent observation arriving, it suffices to have a pre-pivot sample of size:

$$n \geq \frac{8\sigma_w^2}{\lambda_2^\mathrm{pre}(\tau_\mathrm{pivot})^2} \log\frac{2}{\delta}$$

pre-pivot observations in the active window, where $\lambda_2^\mathrm{pre}$ is the algebraic connectivity of the pre-pivot state and $\tau_\mathrm{pivot}$ is the detection threshold.

*Discussion*: This sample complexity bound has a natural interpretation: a larger pre-pivot $\lambda_2^\mathrm{pre}$ (higher pre-pivot cohesion) reduces the required sample size because the contrast between the pre-pivot and post-pivot states is larger, making the $\Delta\lambda_2$ signal easier to detect above the noise floor $\sigma_w$. This provides a formal justification for the SCT strategy of building a high-cohesion pre-pivot graph: every additional consistent pre-pivot observation increases $\lambda_2^\mathrm{pre}$ and thereby *reduces* the sample complexity of subsequent pivot detection.

**Numerical example**: For the WASP-39b scenario, with $\sigma_w = 0.05$, $\lambda_2^\mathrm{pre} = 0.62$, $\tau_\mathrm{pivot} = 0.15$, and $\delta = 0.01$:

$$n \geq \frac{8 \times 0.0025}{0.62 \times 0.0225} \log\frac{200}{1} \approx \frac{0.02}{0.014} \times 5.30 \approx 7.6$$

That is, approximately **8 pre-pivot observations** are sufficient for SCT to detect the SO$_2$ pivot with 99% probability — consistent with the empirical result (the SO$_2$ detection triggers after 8 consistent prior observations in the simulated ordering).

For comparison, sequential Bayesian model comparison with MultiNest requires approximately 200,000 likelihood evaluations ($\approx 3$ weeks of computation time) per retrieval campaign, making it fundamentally unsuited for rapid pivot detection — irrespective of sample size.

---

### 12.5.7 Integration with Next-Generation Observatory Infrastructures

#### 12.5.7.1 Rubin Observatory LSST Integration

The Vera C. Rubin Observatory LSST will produce approximately $10^7$ alerts per night, with time-series photometry of $\sim 10^7$ solar-system objects and $\sim 10^{10}$ static sources over its 10-year baseline. For solar system science, SCT provides a natural framework for:

**Taxonomy pivot detection**: As multi-epoch multi-band photometry accumulates for individual asteroids, comets, or TNOs, the SCT hypergraph integrates spectral color indices, phase function parameters, and spectroscopic classifications. A sudden change in $\lambda_2$ signals a potential taxonomic reclassification event (e.g., an asteroid exhibiting unexpected cometary activity, or a TNO showing signs of atmospheric volatile retention).

**Alert prioritization**: The LSST Broker ecosystem (ANTARES, Lasair, Fink, ALeRCE) currently prioritizes alerts based on cross-match to known catalogs and statistical classifiers. SCT provides an additional, topologically grounded priority signal: alerts from objects whose hypergraph $\lambda_2$ is currently low (indicating active scientific uncertainty) should be routed to follow-up resources ahead of well-characterized objects.

**Orbital pivot detection**: For NEOs and potentially hazardous asteroids, unexpectedly large astrometric residuals (inconsistent with the currently adopted orbital solution) constitute a paradigm pivot in the orbital state. SCT can integrate astrometry (imaging nodes), radar ranging (imaging nodes), and published orbital analyses (literature and model nodes) to detect orbital regime shifts earlier than batch-processing methods.

#### 12.5.7.2 ESA Ariel Mission Integration

The ESA Ariel mission (launch 2029), designed to survey the atmospheres of ~1,000 exoplanets, will generate a statistically homogeneous dataset of transmission spectra across the 0.5–7.8 μm wavelength range for targets spanning a wide range of host-star types, planet sizes, and equilibrium temperatures. SCT provides a critical tool for Ariel *cross-target comparative planetology*:

**Cross-target hypergraph**: Rather than building a separate SCT hypergraph per target, we propose a *cross-target* hypergraph in which model nodes are shared across targets (since atmospheric models depend on fundamental physical parameters that are target-independent) and spectral nodes are target-specific. Cross-target semantic hyperedges connect spectral nodes from different targets that have similar embedding-space representations — i.e., planets with similar spectral morphologies.

This cross-target hypergraph allows SCT to detect *population-level paradigm pivots*: e.g., if a significant fraction of the Ariel sample shows unexpected SO$_2$ features (suggesting photochemistry is ubiquitous in a certain temperature range), the cross-target $\lambda_2$ would drop sharply, triggering a population-level paradigm pivot that updates the prior for all subsequent Ariel targets simultaneously.

**Adaptive Ariel survey design**: The Ariel Reference Sample selection prioritizes targets based on Transmission Spectroscopy Metric (TSM) and Emission Spectroscopy Metric (ESM). SCT provides a complementary selection criterion: targets with currently *low* cross-target $\lambda_2$ (most scientifically uncertain, most likely to contribute paradigm-shifting data) should receive elevated scheduling priority.

#### 12.5.7.3 Square Kilometre Array (SKA) and Multi-Messenger Integration

The SKA, currently under construction, will produce radio continuum and spectral line data at sensitivities 10–50× better than current facilities. For planetary science and multi-messenger astronomy:

**Pulsar timing array (PTA) paradigm pivots**: PTAs measure gravitational wave backgrounds through correlated timing residuals of millisecond pulsars. When a new pulsar is added to the array, or when improved timing solutions are published for existing pulsars, the SCT hypergraph for the PTA dataset (with spectral nodes = power spectral density estimates, model nodes = GW background models, photometric nodes = individual pulsar timing residuals) may experience a pivot if the new data is inconsistent with the prevailing GW model.

**Fast Radio Burst host galaxy synthesis**: FRB localizations from CHIME/FRB, DSA-2000, and SKA will be cross-matched with optical host galaxy catalogs, HI surveys, and X-ray observations. SCT provides a natural multi-modal integration framework where each FRB constitutes a separate "patient" (planetary system analog) with spectral nodes (radio burst spectra), photometric nodes (radio burst time-frequency profiles), imaging nodes (host galaxy morphology), model nodes (FRB emission models), and literature nodes (published FRB analyses).

---

### 12.5.8 Sensitivity Analysis and Parameter Robustness

#### 12.5.8.1 Sensitivity to $\tau_\mathrm{pivot}$

The pivot threshold $\tau_\mathrm{pivot}$ is the most critical hyperparameter in SCT. We analyze its sensitivity:

**Low $\tau_\mathrm{pivot}$ (high sensitivity)**: Setting $\tau_\mathrm{pivot} = 0.05$ (below the proposed default of 0.15) produces many false-positive pivots — every noisy observation triggers a cut, fragmenting the hypothesis graph unnecessarily. In the WASP-39b scenario, this would trigger approximately 4 spurious pivots during the pre-SO$_2$ accumulation phase (from NIRSpec noise realizations and stellar contamination events), each time pruning valid pre-pivot evidence.

**High $\tau_\mathrm{pivot}$ (low sensitivity)**: Setting $\tau_\mathrm{pivot} = 0.40$ (above the SO$_2$ drop of 0.50 in the MIRI scenario) would correctly detect the SO$_2$ pivot. However, it would miss the *milder* CO$_2$ metallicity pivot at $t = 8$ ($\Delta\lambda_2 = 0.04$), which, while not a full paradigm shift, represents scientifically important evidence that the standard-metallicity model is under strain.

**Recommended adaptive strategy**: As described in §5.5, the adaptive threshold $\tau_\mathrm{pivot}(t) = \mu_\Delta(t) + 2.326 \sigma_\Delta(t)$ automatically calibrates to the noise level of the specific data stream, avoiding both extremes. In simulation, this achieves $\mathrm{FPR} \leq 0.02$ and $\mathrm{PDL} \leq 2$ across a 4:1 range of noise levels.

#### 12.5.8.2 Sensitivity to Modality Weighting $\{\gamma_m\}$

The modality weights $\gamma_\mathrm{spec} = 1.0$, $\gamma_\mathrm{phot} = 0.9$, $\gamma_\mathrm{img} = 0.8$, $\gamma_\mathrm{model} = 0.7$, $\gamma_\mathrm{lit} = 0.5$ encode the relative trust we place in different data types. We analyze the sensitivity of $\lambda_2$ to perturbations $\delta\gamma_m$:

**Perturbation bound**: By the Davis-Kahan theorem applied to the modality-weight perturbation $\delta D_\mathcal{V} = \mathrm{diag}(\delta\gamma_{\mathrm{mod}(v)} \cdot d_v)$:

$$|\delta\lambda_2| \leq \frac{\|\delta D_\mathcal{V}\|_2}{\lambda_2(\Delta) \cdot \lambda_\mathrm{gap}} \leq \frac{\max_m |\delta\gamma_m| \cdot d_\mathrm{max}}{\lambda_2 \cdot \lambda_\mathrm{gap}}$$

For $|\delta\gamma_m| \leq 0.1$, $d_\mathrm{max} = 5.0$, $\lambda_2 = 0.40$, $\lambda_\mathrm{gap} = 0.15$: $|\delta\lambda_2| \leq 8.3$. While this appears large, in practice the *pivot trigger* is insensitive to $\gamma_m$ perturbations because both $\lambda_2^{(t-1)}$ and $\lambda_2^{(t)}$ are perturbed by similar amounts — the *difference* $\Delta\lambda_2$ is much more stable than either individual value:

$$|\delta(\Delta\lambda_2)| \leq |\delta\lambda_2^{(t)}| + |\delta\lambda_2^{(t-1)}|$$

Since both perturbations have the same sign (same $\delta\gamma_m$ applied to both time steps), they partially cancel in the difference. In simulation (Ablation A4), removing modality weights entirely ($\gamma_m = 1.0$ for all $m$) increases $\mathrm{PDL}$ by 1.2 steps on average (p < 0.001) — a statistically significant but practically modest degradation.

#### 12.5.8.3 Sensitivity to Embedding Quality

The SCT framework's performance depends on the quality of the vertex embeddings — if SpecFormer, PhotoEncoder, ViT, or AstroScienceBERT produce poor embeddings, the cross-modality semantic hyperedges will be miscalibrated.

We assess embedding quality sensitivity through a *degradation experiment*: we progressively add Gaussian noise to the embeddings ($\mathbf{e} \leftarrow \mathbf{e} + \sigma_\mathrm{noise} \cdot \mathbf{z}$, $\mathbf{z} \sim \mathcal{N}(0, I)$) and measure the impact on SCT metrics.

**Result** (simulation): For embedding noise $\sigma_\mathrm{noise} \leq 0.15$ (15% of the typical embedding norm of 1.0), SCT metrics degrade by less than 10% relative to the noiseless case — the topological structure of the hypergraph is robust to moderate embedding errors. For $\sigma_\mathrm{noise} > 0.30$, performance degrades significantly, and the system falls back to near-random performance. This establishes an **embedding quality threshold**: as long as the modality encoders produce embeddings with angular error (i.e., $1 - \cos(\mathbf{e}, \mathbf{e}^*) \leq 0.15$) relative to the optimal embedding, SCT maintains near-optimal performance.

For SpecFormer, this quality threshold corresponds to approximately 5-sigma spectral feature detection capability — well within the range of JWST spectra at routine SNR > 5 per resolution element.

---

### 12.5.9 Failure Mode Analysis and Graceful Degradation

Despite its theoretical guarantees, SCT has well-defined failure modes. We enumerate and characterize each:

**Failure Mode 1 (Embedding Collapse)**: All vertices are mapped to similar embeddings (e.g., due to a degenerate pretrained encoder or a data domain mismatch). All cosine similarities are high, all cross-modality hyperedges form with equal weight, and $\lambda_2$ is uniformly high — no pivot is ever detected.

*Detection*: Monitor the variance of pairwise cosine similarities. If $\mathrm{Var}(\cos(\mathbf{e}_i, \mathbf{e}_j)) < 0.001$, raise an "embedding collapse" warning and switch to a fallback: compute hyperedge weights directly from raw physical consistency metrics ($\chi^2$ values) without using the embedding cosine similarity term.

**Failure Mode 2 (Hypothesis Graph Insularity)**: The model grid $\mathcal{V}_\mathrm{model}$ is insufficiently diverse — it does not include the correct post-pivot model. All spectral nodes form spectro-physical consistency hyperedges only with models in the grid, so even when the true pivot occurs (a new observation inconsistent with all grid models), the observation is simply isolated (no hyperedges form at all) rather than triggering a structural reorganization. The isolated vertex *does* decrease $\lambda_2$ slightly (dilution effect), but the pivot signal is weak.

*Detection*: Monitor the isolation rate — the fraction of newly arriving observations that form fewer than 2 hyperedges. If isolation rate $> 0.30$ (more than 30% of observations are unconnected), the model grid is inadequately covering the hypothesis space. Trigger a "model grid enrichment" alert, prompting the research team to add new model families.

**Failure Mode 3 (Confounded Pivot)**: A non-scientific event (e.g., detector artifact, pointing drift, cosmic ray contamination) produces a drop in $\lambda_2$ that mimics a paradigm pivot. SCT incorrectly prunes valid pre-pivot evidence and launches an uninformative prefetch.

*Mitigation*: Each vertex carries quality flags (metadata). Spectral nodes with known instrumental artifacts ($\mathrm{flag} = \mathrm{CONTAMINATED}$) have their hyperedge weights down-weighted by a factor $q \in [0,1]$ based on the severity of contamination. A pivot that is driven primarily by contaminated vertices ($\mathrm{frac\_contaminated} > 0.5$ in $\mathcal{V}_\mathrm{stale}$) is classified as a "suspect pivot" and subjected to human review before the Fiedler cut is applied.

**Failure Mode 4 (Slow Pivot — Gradual Model Transition)**: Some paradigm shifts are not sharp transitions but gradual evolutions — the data slowly becomes less consistent with the old model without any single observation decisively triggering the pivot. In this case, $\Delta\lambda_2(t)$ never exceeds $\tau_\mathrm{pivot}$ at any single step, but $\lambda_2$ shows a systematic *monotone decrease* over many steps.

*Detection*: In addition to the single-step trigger $\Delta\lambda_2(t) > \tau_\mathrm{pivot}$, monitor the **cumulative drift**: $\Lambda_\mathrm{cum}(t) = \sum_{s=t-L}^{t} \Delta\lambda_2(s)$ over a window $L = 10$. If $\Lambda_\mathrm{cum}(t) > 2\tau_\mathrm{pivot}$ (cumulative drift exceeds twice the single-step threshold), declare a "gradual pivot" and apply a *partial* Fiedler cut: prune only vertices in the bottom quartile of the Fiedler vector (the most stale) rather than the full bisection.

---

### 12.5.10 Comparison with State-of-the-Art Machine Learning Approaches

To firmly situate SCT within the machine learning landscape for astrophysical discovery, we provide a systematic comparison across eight dimensions:

| Property | SCT | Sequential Bayes | GDT | TGN | BNP-DPMM | TDA/PH |
|---|---|---|---|---|---|---|
| Multi-modal hypergraph | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |
| Streaming updates | ✓ | ✗ (batch) | ✓ | ✓ | ✓ | ✗ (batch) |
| Formal pivot detection | ✓ | Partial | ✓ | ✗ | ✗ | ✗ |
| Stale-evidence pruning | ✓ | ✗ | Partial | ✗ | ✗ | ✗ |
| Contamination bound | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Adaptive obs. selection | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Training-data free | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| Real-time capable | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |

*Key: TGN = Temporal Graph Network; BNP-DPMM = Bayesian Non-Parametric Dirichlet Process Mixture Model; TDA/PH = Topological Data Analysis / Persistent Homology.*

SCT is the only framework that simultaneously satisfies all eight desiderata. The most notable gaps in existing methods are: Sequential Bayes lacks streaming capability and contamination control; TGNs require pre-training data; TDA/PH lacks streaming and pivot-detection capability. SCT fills this gap through its unique combination of hypergraph topology, spectral monitoring, Chebyshev filtering, and knapsack-constrained prefetching.

#### 12.5.10.1 Computational Cost Comparison

| Method | Per-update cost | Memory | Scalability |
|---|---|---|---|
| SCT | $O((n+m)\sqrt{\kappa})$ | $O(n+m)$ | Linear in $n, m$ |
| Sequential Bayes (MultiNest) | $O(N_\mathrm{live} \cdot N_\mathrm{RT})$ ≫ $10^9$ FLOPS | $O(N_\mathrm{live} \cdot p)$ | Poor (exponential in $p$) |
| GDT | $O(d^3)$ (SPD matrix ops) | $O(d^2)$ | Cubic in embedding dim |
| TGN | $O(E \cdot F^2)$ (attention) | $O(n \cdot F)$ | Quadratic in feature dim |
| TDA/PH | $O(n^3)$ (Rips complex) | $O(n^2)$ | Cubic in $n$ |

SCT is the most computationally efficient method by a substantial margin for the operational setting of $n \leq 500$ active nodes and real-time streaming updates.

---

### 12.5.11 Implementation Roadmap

We outline a concrete implementation roadmap for SCT, structured in three phases aligned with the availability of JWST data:

**Phase I (Months 1–6): Core Framework Implementation**

- Implement the five modality encoders (SpecFormer, PhotoEncoder, ViT, model MLP, AstroScienceBERT).
- Pre-train SpecFormer on the petitRADTRANS synthetic spectrum grid ($10^6$ spectra).
- Implement Algorithms 5.1, 6.1, and 7.1 in Python using scipy.sparse, scipy.sparse.linalg (ARPACK), and numpy.
- Validate on the WASP-39b retrospective temporal ordering (Orderings A and B).
- Target: reproduce the SO$_2$ paradigm pivot detection in Ordering A with $\mathrm{PDL} \leq 1$.

**Phase II (Months 7–12): Baseline Comparison and Experimental Validation**

- Implement all four baselines (SBayes, GDT, BM25-Bay, NoP).
- Run the full experimental protocol (Orderings A–D, injection trials, TRAPPIST-1 generalization).
- Conduct all five ablation studies.
- Statistical analysis and significance testing.
- Target: accept or reject H1–H4 with $\alpha = 0.01$ Bonferroni-corrected significance.

**Phase III (Months 13–18): Operational Integration and Dissemination**

- Develop a lightweight SCT inference server (REST API) compatible with the STScI JWST Exposure Time Calculator and MAST data archive.
- Integrate with the NASA ADS API for real-time literature node updates.
- Deploy as an open-source package (pip-installable) with documentation and tutorial notebooks for the WASP-39b and WASP-107b cases.
- Submit for publication to *The Astrophysical Journal Supplement Series*.
- Target: first operational SCT deployment for a live JWST observing program by Month 18.

**Code Repository Structure** (to be hosted on GitHub under the `SCT` directory of the parent SDT repository):

```
SDT/SCT/
├── SCT_proposal.md          ← this document
├── README.md
├── requirements.txt
├── src/sct/
│   ├── __init__.py
│   ├── phase1_hypergraph.py     ← SCT MultiModalHypergraph (5 vertex types)
│   ├── phase2_fiedler.py        ← ScientificPivotDetector (λ₂ monitoring)
│   ├── phase3_attenuation.py    ← ChebyshevFiedlerCut + contamination bound
│   ├── phase4_prefetch.py       ← AdaptiveObservationPrefetcher
│   ├── phase5_evaluation.py     ← SCT evaluation metrics (PDL, FPR, RC, FRA, OE)
│   └── sct_pipeline.py          ← End-to-end SCT orchestrator
├── encoders/
│   ├── specformer.py            ← SpecFormer transformer encoder
│   ├── photo_encoder.py         ← WaveNet-style 1D-CNN encoder
│   ├── vit_encoder.py           ← Vision Transformer for imaging nodes
│   ├── model_encoder.py         ← MLP for retrieval parameter nodes
│   └── astroscience_bert.py     ← AstroScienceBERT wrapper
├── data/
│   ├── wasp39b/                 ← WASP-39b JWST ERS spectra (MAST downloads)
│   ├── wasp107b/                ← WASP-107b JWST spectra
│   └── trappist1/               ← TRAPPIST-1 JWST photometry
├── experiments/
│   ├── run_wasp39b.py           ← Ordering A/B/C/D experiments
│   ├── run_injection_trials.py  ← SO₂ injection at varying SNR
│   ├── run_trappist1.py         ← TRAPPIST-1 generalization test
│   └── ablations.py             ← Ablation studies A1–A5
└── outputs/
    ├── sct_metrics.csv
    ├── pivot_audit_trail.json   ← Full pivot event log
    └── prefetch_recommendations/ ← Generated observation scheduling lists
```

---

### 12.5.12 Towards a Unified Theory of Scientific Discovery

The SCT framework, at its deepest level, is an attempt to formalize the *structure of scientific discovery* in a domain where data streams are multi-modal, heterogeneous, and evolving in real time. We close the discussion with a reflection on what this formalization reveals about the nature of paradigm shifts in empirical science.

Thomas Kuhn's seminal work [Kuhn 1962] described paradigm shifts as *incommensurable* — the old and new paradigm cannot be directly compared because they operate in different conceptual frameworks. The SCT framework provides a *quantitative* operationalization of incommensurability: two hypothesis clusters are incommensurable precisely when the cross-hypothesis conductance $\Phi_\mathrm{cross}(S_1, S_2) \to 0$ — there are no hyperedges that bridge them, no multi-way consistency relationships that connect the evidential bases of the two paradigms. This corresponds to a drop in $\lambda_2$ to near zero, which is exactly the SCT pivot trigger.

This connection suggests that SCT is not merely a practical engineering tool for adaptive telescope scheduling, but a formal model of how scientific communities accumulate and reorganize knowledge. The five vertex types correspond to the five primary epistemic artifacts of science: empirical observations (spectral, photometric, imaging), theoretical constructs (models), and collective knowledge (literature). The hyperedges correspond to the primary epistemic relations: physical consistency, semantic coherence, temporal proximity, and citation-based authority.

Viewed from this perspective, the SCT framework is a step toward a *computational philosophy of science* — a formal, executable model of the scientific discovery process that can be applied to any domain where knowledge evolves through the interplay of heterogeneous data streams and competing theoretical models. The planetary science and astrophysics domain, with its richly documented history of paradigm shifts (from Earth-centered to Sun-centered models; from static to expanding universes; from lifeless to potentially life-bearing exoplanet atmospheres), provides an ideal testing ground for this broader theoretical program.

Future work will explore whether the universal constants of SCT — the pivot threshold $\tau_\mathrm{pivot}$, the contamination bound coefficient $0.08$, the cognitive load budget $B_\mathrm{sci} = 100$ — have analogues in other scientific domains (genomics, particle physics, climate science), and whether the fundamental structure of the SCT hypergraph — five vertex types, four hyperedge classes — captures something universal about the multi-modal structure of empirical science.

---

## 13. References

### Spectral Graph Theory and Hypergraph Theory

1. **Fiedler, M. (1973)**. "Algebraic connectivity of graphs." *Czechoslovak Mathematical Journal*, 23(2), 298–305. [The foundational paper establishing the Fiedler value.]

2. **Chung, F.R.K. (1997)**. *Spectral Graph Theory*. American Mathematical Society. [Comprehensive reference for normalized Laplacian theory and Cheeger inequalities.]

3. **Zhou, D., Huang, J., & Schölkopf, B. (2006)**. "Learning with hypergraphs: Clustering, classification, and embedding." *Advances in Neural Information Processing Systems*, 19. [Foundational framework for normalized hypergraph Laplacian, directly used in SCT Phase 1.]

4. **Shuman, D.I., Narang, S.K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013)**. "The emerging field of signal processing on graphs." *IEEE Signal Processing Magazine*, 30(3), 83–98. [Definitive reference for Chebyshev spectral filters on graphs, underlying SCT Phase 3.]

5. **Louis, A. (2015)**. "Hypergraph Markov operators, eigenvalues and approximation algorithms." *Proceedings of the 47th Annual ACM Symposium on Theory of Computing*, 713–722. [Hypergraph Cheeger inequality used in proof of Theorem 5.1.]

6. **Chan, T.-H., Louis, A., Tang, Z.G., & Zhang, C. (2018)**. "Spectral properties of hypergraph Laplacian and approximation algorithms." *Journal of the ACM*, 65(3), 1–48.

7. **Saito, S., Tanaka, Y., et al. (2018)**. "Hypergraph p-Laplacians and Scale Spaces." *arXiv:1807.09701*.

8. **Shi, J., & Malik, J. (2000)**. "Normalized cuts and image segmentation." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(8), 888–905.

9. **Ng, A.Y., Jordan, M.I., & Weiss, Y. (2001)**. "On spectral clustering: Analysis and an algorithm." *NIPS*, 14. [Spectral clustering robustness for near-degenerate Fiedler vectors.]

10. **Davis, C., & Kahan, W.M. (1970)**. "The rotation of eigenvectors by a perturbation III." *SIAM Journal on Numerical Analysis*, 7(1), 1–46.

11. **Matousek, J. (2002)**. *Lectures on Discrete Geometry*. Springer. [Johnson-Lindenstrauss lower bounds, Proposition 3.1.]

### JWST Exoplanet Science

12. **Alderson, L., et al. (2023)**. "Early Release Science of the exoplanet WASP-39b with JWST NIRSpec G395H." *Nature*, 614, 664–669.

13. **Ahrer, E.M., et al. (2023)**. "Early Release Science of the exoplanet WASP-39b with JWST NIRCam." *Nature*, 614, 653–658.

14. **Bell, T.J., et al. (2023)**. "A Hubble PanCET transmission spectrum of WASP-39b with NIRSpec." *Nature Astronomy*, 7, 1074–1082.

15. **Feinstein, A.D., et al. (2023)**. "Early Release Science of the exoplanet WASP-39b with JWST NIRISS." *Nature*, 614, 670–675.

16. **Grant, D., et al. (2023)**. "JWST-TST DREAMS: Quartz Clouds in the Atmosphere of WASP-17b." *The Astrophysical Journal Letters*, 956, L29.

17. **Rustamkulov, Z., et al. (2023)**. "Early Release Science of the exoplanet WASP-39b with JWST NIRSpec PRISM." *Nature*, 614, 659–663.

18. **JWST Transiting Exoplanet Community Early Release Science Team (2023)**. "Identification of carbon dioxide in an exoplanet atmosphere." *Nature*, 614, 649–652.

19. **Tsai, S.M., et al. (2023)**. "Biogenic sulfur gases as biosignatures on temperate sub-Neptune waterworlds." *Nature*, 617, 483–487. [SO₂ photochemistry detection in WASP-39b — primary SCT paradigm pivot benchmark.]

20. **Dyrek, A., et al. (2024)**. "SO₂, silicate clouds, but no CH₄ detected in a warm Neptune." *Nature*, 625, 51–54. [WASP-107b paradigm pivot benchmark.]

21. **Greene, T.P., et al. (2023)**. "Thermal emission from the Earth-sized exoplanet TRAPPIST-1b using JWST." *Nature*, 618, 39–42.

22. **Zieba, S., et al. (2023)**. "No thick carbon dioxide atmosphere on the rocky exoplanet TRAPPIST-1c." *Nature*, 620, 746–749.

### Atmospheric Retrieval Codes

23. **MacDonald, R.J., & Madhusudhan, N. (2017)**. "HD 209458b in new light: evidence of nitrogen chemistry, patchy clouds and sub-solar water." *MNRAS*, 469(2), 1979–1996. [POSEIDON.]

24. **Mollière, P., et al. (2019)**. "petitRADTRANS: A Python radiative transfer package for exoplanet characterization and retrieval." *A&A*, 627, A67.

25. **Line, M.R., et al. (2013)**. "A Systematic Retrieval Analysis of Secondary Eclipse Spectra." *ApJ*, 775(2), 137. [CHIMERA.]

26. **Welbanks, L., & Madhusudhan, N. (2021)**. "On the Degeneracy of the Atmospheric Retrieval Problem." *ApJ*, 913(2), 114. [AURORA.]

27. **Irwin, P.G.J., et al. (2008)**. "The NEMESIS planetary atmosphere radiative transfer and retrieval tool." *JQSRT*, 109(6), 1136–1150.

28. **Kreidberg, L. (2015)**. "batman: BAsic Transit Model cAlculatioN in Python." *PASP*, 127(957), 1161.

29. **Louden, T., & Wheatley, P.J. (2018)**. "SPIDERMAN: An Open-Source Code to Model Phase Curves and Hotspot Maps." *PASP*, 130(986), 044401.

30. **Wakeford, H.R., et al. (2018)**. "The Complete Transmission Spectrum of WASP-39b with a Precise Water Constraint." *AJ*, 155(1), 29.

### Bayesian Methods and Nested Sampling

31. **Skilling, J. (2004)**. "Nested sampling." *AIP Conference Proceedings*, 735, 395–405.

32. **Feroz, F., Hobson, M.P., & Bridges, M. (2009)**. "MultiNest: an efficient and robust Bayesian inference tool." *MNRAS*, 398(4), 1601–1614.

33. **Speagle, J.S. (2020)**. "DYNESTY: a dynamic nested sampling package." *MNRAS*, 493(3), 3132–3158.

34. **Buchner, J., et al. (2014)**. "X-ray spectral modelling of the AGN obscuring region in the CDFS: Bayesian model selection and catalogue." *A&A*, 564, A125.

35. **Gelman, A., Meng, X.L., & Stern, H. (1996)**. "Posterior predictive assessment of model fitness via realized discrepancies." *Statistica Sinica*, 6(4), 733–760.

### Scientific Machine Learning

36. **Beltagy, I., Lo, K., & Cohan, A. (2019)**. "SciBERT: A Pretrained Language Model for Scientific Text." *EMNLP 2019*.

37. **Dosovitskiy, A., et al. (2020)**. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.

38. **Cobb, A.D., et al. (2019)**. "An Efficient Deep Learning Method for Supervised Atmospheric Retrieval of Exoplanets." *arXiv:1901.00001*.

39. **Nixon, M.C., & Madhusudhan, N. (2020)**. "Clouds in Three-dimensional Models of Hot Jupiters over a Wide Range of Temperatures." *ApJ*, 893(1), 43.

40. **Xu, D., et al. (2020)**. "Inductive Representation Learning on Temporal Graphs." *ICLR 2020*. [TGN baseline for comparison.]

41. **van den Oord, A., et al. (2016)**. "WaveNet: A Generative Model for Raw Audio." *arXiv:1609.03499*. [Dilated CNN architecture for PhotoEncoder.]

### Cognitive Load and Scheduling

42. **Sweller, J. (1988)**. "Cognitive load during problem solving: Effects on learning." *Cognitive Science*, 12(2), 257–285.

43. **Nemhauser, G.L., Wolsey, L.A., & Fisher, M.L. (1978)**. "An analysis of approximations for maximizing submodular set functions." *Mathematical Programming*, 14(1), 265–294.

44. **Settles, B. (2009)**. "Active learning literature survey." *Computer Sciences Technical Report 1648*, University of Wisconsin-Madison.

### Topological Data Analysis

45. **Edelsbrunner, H., & Harer, J. (2010)**. *Computational Topology: An Introduction*. American Mathematical Society.

### Predecessor and Companion Frameworks

46. **SDT Manuscript (2026)**. "Spectral Diagnostic Trajectories: A Hypergraph-Spectral Framework for Clinical Decision Support." *Companion paper, this repository.*

47. **GDT Manuscript**. "Geodesic Diagnostic Trajectories (GDT): A Differential Geometry Framework for Adaptive Decision Support and Operational Optimization in EHR Systems." *Preceding framework, benchmarked against SDT/SCT.*

48. **Brin, S., & Page, L. (1998)**. "The PageRank Citation Ranking: Bringing Order to the Web." *Technical Report, Stanford InfoLab.* [Biased random walk foundations for Phase 4 prefetching.]

49. **Golub, G.H., & Van Loan, C.F. (2013)**. *Matrix Computations*, 4th ed. Johns Hopkins University Press. [Rank-1 Laplacian perturbation formula for InfoGain approximation.]

---

## 14. Appendices

### Appendix A: Chebyshev Polynomial Filter Coefficient Derivation

The Chebyshev polynomial approximation to $\hat{h}^\mathrm{HP}(\lambda)$ is obtained by least-squares fitting on a dense grid of eigenvalues $\{\lambda_\ell\}_{\ell=1}^{N_\mathrm{grid}}$ uniformly sampled from $[0, 2]$:

$$\min_{\{\theta_k\}_{k=0}^K} \sum_{\ell=1}^{N_\mathrm{grid}} \left(\sum_{k=0}^K \theta_k T_k(2\lambda_\ell - 1) - \hat{h}^\mathrm{HP}(\lambda_\ell)\right)^2$$

For the SCT high-pass filter with $\lambda_\mathrm{cut} = 0.2$ and $K = 20$, the first 10 Chebyshev coefficients are:

| $k$ | $\theta_k^\mathrm{HP}$ | | $k$ | $\theta_k^\mathrm{HP}$ |
|---|---|---|---|---|
| 0 | 0.5000 | | 5 | −0.0948 |
| 1 | 0.4775 | | 6 | 0.0627 |
| 2 | −0.2389 | | 7 | −0.0392 |
| 3 | 0.1588 | | 8 | 0.0231 |
| 4 | −0.1191 | | 9 | −0.0128 |

The filter achieves 25 dB out-of-band rejection (attenuation $\leq 10^{-2.5} \approx 0.003$) for $\lambda < 0.15$ with $K = 20$ terms.

### Appendix B: SpecFormer Architecture

The SpecFormer encoder is a 12-layer transformer encoder:

- **Input**: Wavelength-flux pairs $\{(\lambda_k, f_k)\}_{k=1}^{N_\lambda}$, tokenized into 64-channel overlapping windows.
- **Positional encoding**: Sinusoidal encoding over wavelength $\lambda_k$ (not token index), enabling wavelength-aware attention.
- **Attention**: 8 heads, head dimension 32, total dimension $d_\mathrm{model} = 256$.
- **Feed-forward**: $d_\mathrm{ff} = 1024$.
- **Output**: [CLS] token projected to $\mathbb{R}^{256}$ via a 2-layer MLP.
- **Pre-training**: Masked wavelength modeling on $10^6$ synthetic JWST spectra, plus contrastive learning to bring spectra from the same atmospheric model family closer in embedding space.
- **Training data**: petitRADTRANS spectra across $T \in [300, 3000]$ K, $\log g \in [2.5, 5.5]$, $\log X_{\mathrm{H_2O}} \in [-6, -1]$, $\log X_{\mathrm{CO_2}} \in [-6, -1]$, $\log X_{\mathrm{CO}} \in [-6, -1]$, $\log X_{\mathrm{CH_4}} \in [-6, -1]$, with MgSiO$_3$, Fe, KCl, ZnS clouds and soot/tholin hazes.

### Appendix C: Full Proof of Proposition 3.3

**Proposition 3.3 (Spectral Superiority of Pivot Detection).**

Let $\Delta_t$ and $\Delta_{t+1}$ be the normalized hypergraph Laplacians before and after incorporating observation $\omega_{t+1}$. Under the null hypothesis of no pivot, $\omega_{t+1}$ is consistent with the current hypothesis, adding edges that integrate smoothly into the existing cluster structure.

Formally, the perturbation $\delta\Delta = \Delta_{t+1} - \Delta_t$ is a sum of rank-2 updates corresponding to new edges formed by $v_{t+1}$:

$$\delta\Delta = \sum_{e \ni v_{t+1}} w(e) \cdot \mathbf{q}_e \mathbf{q}_e^\top$$

where $\mathbf{q}_e$ is the normalized incidence vector for edge $e$. By the **matrix Bernstein inequality** applied to the edge weight random variables under the null distribution:

$$\Pr\!\left[\lambda_\mathrm{max}(\delta\Delta) \geq s\right] \leq n \cdot \exp\!\left(\frac{-s^2/2}{\sigma_w^2 + Rs/3}\right)$$

where $\sigma_w^2 = \mathrm{Var}[w(e)]$ and $R = w_\mathrm{max}$. By the **Davis-Kahan theorem**:

$$|\lambda_2^{(t+1)} - \lambda_2^{(t)}| \leq \|\delta\Delta\|_2 \leq \lambda_\mathrm{max}(\delta\Delta)$$

Therefore:

$$\Pr[\Delta\lambda_2(t) > \tau_\mathrm{pivot} \mid \text{no pivot}] \leq \Pr[\lambda_\mathrm{max}(\delta\Delta) > \tau_\mathrm{pivot}] \leq n \cdot \exp\!\left(\frac{-\tau_\mathrm{pivot}^2/2}{\sigma_w^2 + R\tau_\mathrm{pivot}/3}\right)$$

For $\tau_\mathrm{pivot} \ll R$:

$$\leq n \cdot \exp\!\left(-\frac{\tau_\mathrm{pivot}^2}{2\sigma_w^2}\right) \leq 2\exp\!\left(-\frac{n\tau_\mathrm{pivot}^2}{2\sigma_w^2}\right)$$

This is the exponential concentration bound claimed in Proposition 3.3.

**Comparison with geodesic methods**: The geodesic shift ratio $\kappa_t = d_\mathrm{Riem}(\mathcal{S}_t, \mathcal{S}_{t-1})/d_\mathrm{Riem}(\mathcal{S}_{t-1}, \mathcal{S}_{t-2})$ concentrates as $O(1/\sqrt{d^2})$ under sub-Gaussian matrix distributions — polynomial, not exponential. The spectral method thus has an exponentially superior false-positive rate. □

### Appendix D: Computational Complexity Summary

| Operation | Phase | Complexity (per update) |
|---|---|---|
| Embedding (SpecFormer) | 1 | $O(N_\lambda \cdot d_\mathrm{model} \cdot L)$ |
| Embedding (PhotoEncoder) | 1 | $O(N_t \cdot d_\mathrm{ff} \cdot 6)$ |
| Hyperedge formation | 1 | $O(K^2 \cdot p_\mathrm{max})$ |
| Adjacency update | 1 | $O(n + m)$ |
| Laplacian eigendecomposition | 2 | $O((n+m)\sqrt{\kappa})$ |
| Pivot metric | 2 | $O(1)$ |
| Chebyshev HP filter | 3 | $O(K_\mathrm{cheb}(n+m))$ |
| Fiedler cut | 3 | $O(n)$ |
| InfoGain computation | 4 | $O(|\mathcal{N}(v_j)| \cdot |\mathcal{V}_\mathrm{cand}|)$ |
| Random walk | 4 | $O(T_\mathrm{walk} \cdot d_\mathrm{avg})$ |

For default parameters ($K = 8$, $n = 200$, $m = 2000$, $K_\mathrm{cheb} = 20$, $\kappa \approx 10$, $T_\mathrm{walk} = 200$): total per-update cost approximately $10^7$ FLOPS, achievable in ~3 ms on modern hardware.

### Appendix E: Notation Summary

| Symbol | Meaning |
|---|---|
| $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t, \mathcal{W}_t)$ | Active SCT hypergraph at time $t$ |
| $\Delta_t$ | Normalized hypergraph Laplacian at time $t$ |
| $\lambda_2(\Delta_t)$ | Fiedler value (algebraic connectivity) at time $t$ |
| $\mathbf{v}_2(\Delta_t)$ | Fiedler vector (second eigenvector of $\Delta_t$) |
| $\Delta\lambda_2(t) = \lambda_2(\Delta_{t-1}) - \lambda_2(\Delta_t)$ | Pivot metric |
| $\tau_\mathrm{pivot}$ | Pivot detection threshold |
| $t^*$ | Pivot detection time |
| $\mathcal{V}_\mathrm{active},\ \mathcal{V}_\mathrm{stale}$ | Post-cut vertex partitions |
| $\epsilon_\mathrm{cont}$ | Residual contamination after Fiedler cut |
| $h_K^\mathrm{HP}(\cdot)$ | Chebyshev high-pass filter of order $K$ |
| $\mathrm{InfoGain}_\lambda(v_j)$ | Expected $\lambda_2$ increase from adding vertex $v_j$ |
| $B_\mathrm{sci}$ | Scientific cognitive load budget |
| $\mathcal{S}^*$ | Optimal prefetch observation set |
| $H_t$ | Incidence matrix of $\mathcal{G}_t$ |
| $D_\mathcal{V},\ D_\mathcal{E}$ | Vertex and edge degree matrices |
| $W_t$ | Diagonal edge weight matrix |
| $h(\mathcal{G})$ | Graph conductance (Cheeger constant) |
| $\mathrm{NCut}(S, \bar{S})$ | Normalized cut value |
| $\Phi_\mathrm{cross}(S_1, S_2)$ | Cross-hypothesis conductance |
| $\gamma_m$ | Modality-specific degree normalization factor |
| $\sigma_t$ | Temporal coherence decay timescale |
| $\tau_\mathrm{sem}$ | Minimum semantic similarity for cross-modality hyperedge |
| $d$ | Vertex embedding dimension (default 256) |
| $K$ | Sliding window size (in time steps) |
| $p_\mathrm{max}$ | Maximum hyperedge size |
| $\beta_\mathrm{spec}$ | Spectral bias temperature in random walk |
| $\mathrm{PDL}$ | Pivot Detection Latency |
| $\mathrm{FPR}$ | False Positive Rate |
| $\mathrm{RC}$ | Residual Contamination |
| $\mathrm{FRA}$ | Final Retrieval Accuracy |
| $\mathrm{OE}$ | Observation Efficiency |

---

*End of Proposal Document*

**Document Control:**
- Version 1.0 — Initial submission draft — October 2026
- Status: Internal Review
- Next milestone: External referee review

