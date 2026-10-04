Expanded Research Plan: Spectral Diagnostic Trajectories (SDT)

Executive Summary

This document outlines the next evolutionary phase of the Geodesic Diagnostic Trajectories (GDT) framework. While GDT successfully models clinical intent as a continuous trajectory on the Riemannian manifold of Symmetric Positive Definite (SPD) matrices $\mathcal{S}_d^+$, its reliance on sliding-window covariance bounds its ability to natively ingest highly unstructured, continuous, and multi-modal physiological streams.

The Spectral Diagnostic Trajectories (SDT) framework transitions the state space from a Riemannian manifold to a dynamic multi-modal hypergraph. It replaces the geodesic shift ratio $\kappa_t$ with spectral gap analysis (Fiedler value $\lambda_2$) for pivot detection, and replaces JEPA-driven latent prefetching with spectrally-biased random walks over the global EHR graph.

Phase 1: Hypergraph Construction & Multimodal State Space

Objective: Encode the physician's cognitive path and the patient's physical state into a unified, continuous hypergraph topology.

Node Initialization:

Textual & Semantic Nodes: Ingest clinical notes, query logs, and discrete EHR events. Embed using ClinicalBERT (projected to $\mathbb{R}^d$), preserving the semantic geometry established in GDT.

Physiological Nodes: Expand the state space to ingest continuous data (e.g., MIMIC-IV PPG-ECG waveforms, HiRID continuous vitals). These nodes represent high-frequency physiological states encoded via 1D-CNN or state-space models (e.g., Mamba).

Hyperedge Formation:

Unlike GDT's pairwise covariance, hyperedges can connect multiple nodes simultaneously. Let the active subgraph be $G_t = (V_t, E_t)$.

Edges are weighted based on a combination of the clinician's temporal clickstream sequences (sequential proximity) and cosine similarity in the latent embedding space.

State Tracking:

The instantaneous diagnostic hypothesis is no longer an SPD matrix $S_t$, but a continuously evolving active subgraph $G_t$ moving over a sliding temporal window of recent clinician queries and patient data updates.

Phase 2: Spectral Pivot Detection & Generative Dynamics

Objective: Utilize spectral graph theory to detect abrupt diagnostic pivots, replacing the differential-geometric shift ratio $\kappa_t$.

Laplacian Monitoring:

At each time step $t$, compute the normalized graph Laplacian matrix for the active subgraph $G_t$:


$$L_t = I - D_t^{-1/2} A_t D_t^{-1/2}$$


where $A_t$ is the adjacency matrix and $D_t$ is the degree matrix.

Fiedler Tracking:

Monitor the spectrum of $L_t$, specifically the second smallest eigenvalue (the Fiedler value) $\lambda_2(L_t)$.

Diagnostic Interpretation: $\lambda_2$ quantifies the algebraic connectivity (structural cohesion) of the hypothesis. Steady growth indicates diagnostic refinement and evidence accumulation (uncertainty decay).

Threshold Gating:

A sudden drop in $\lambda_2$ indicates the graph is splitting into distinct clusters—a mathematical manifestation of a diagnostic pivot (e.g., shifting from sepsis protocol to renal dosing).

Trigger the interference gate when $\Delta \lambda_2 > \tau_{pivot}$, functionally replacing GDT's Riemannian geodesic jump trigger $\kappa_t > \kappa_0$.

Phase 3: Graph-Signal Context Attenuation

Objective: Bound context contamination and prevent stale hypotheses from polluting downstream trajectory simulation.

Spectral Filtering:

Upon trigger activation ($\Delta \lambda_2 > \tau_{pivot}$), apply a high-pass graph filter. Let $x$ represent the signal of contextual relevance across nodes. The filtered signal $\tilde{x} = H(L_t)x$ actively suppresses low-frequency components (associated with the pre-pivot stale context).

Subgraph Decoupling:

Execute a Fiedler cut using the eigenvector associated with $\lambda_2$. Mathematically sever the hyperedges between the pre-pivot and post-pivot clusters.

This enforces a strict upper bound on information interference, directly advancing GDT's Theorem 3.2 (Curvature-Gated Contamination Decay) into graph space.

Embedding Formulation:

Aggregate the post-pivot nodes to generate a cleanly filtered, focused retrieval vector. This vector serves as the seed for generating the next immediate query state, free of latent-context pollution.

Phase 4: Anticipatory Prefetch Engine & Simulation Interface

Objective: Evolve the retrieval engine into a predictive world model driven by optimal control constraints.

Optimal Control Formulation:

Similar to GDT, frame prefetching to minimize expected clinical decision delay, but natively operate on graph topologies.

The loss function $J(u)$ will balance relevance, structural graph continuity, and human cognitive load bounds.

Biased Traversal:

Replace the JEPA-based target network with a parameterized random walk on the global EHR graph $G_{global}$.

The transition probabilities $P(v_j \vert{} v_i)$ are biased by the active subgraph's spectral density. This allows the system to simulate branching physiological futures and forecast the next clinical intent seamlessly.

Cognitive Gating:

Adopt the cognitive safety principles of GDT (mitigating premature closure and automation bias).

Limit the total volume of prefetched nodes based on transition probability confidence $\sum \text{confidence} < C_{max}$. This strictly respects human cognitive-memory constraints (Sweller's cognitive-load theory) and prevents UI overwhelming.

Phase 5: Evaluation & Operational Analytics (Next Steps)

Objective: Validate SDT using the established benchmark methodology from the GDT study.

Data Sources: Re-use the multi-domain MIMIC-IV, MIMIC-CXR, and HiRID datasets, but actively integrate the continuous MIMIC-IV PPG-ECG waveform metadata for the multimodal node initialization (Phase 1).

Crossover Benchmark: Conduct a randomized crossover evaluation comparing SDT against the original GDT framework, CMA, and BM25 to measure time-to-correct-information and task completion rate.

Queueing Model Translation: Apply the resulting retrieval latency reductions to the $M/M/c$ queueing model of Emergency Department (ED) operations to calculate downstream impacts on queueing wait ($W_q$), length-of-stay (LOS), and bed-turnover throughput.