Phase 2: Pivot Detection & The Interference Gate (GDT to SDT Evolution)

Objective: Mathematically detect abrupt changes in diagnostic intent to trigger the interference gate, transitioning from GDT's differential geometry to SDT's spectral graph theory.

The Legacy GDT Geodesic Trigger

In the foundational GDT model, sudden cognitive pivots (e.g., a clinician shifting focus from respiratory distress to acute kidney injury) were detected by measuring the acceleration of the state matrix $S_t$ across the Riemannian manifold.

Geodesic Distance: The distance between consecutive states is measured as:


$$d_g(S_{t-1}, S_t) = \vert{}\vert{}\log(S_{t-1}^{-1/2}S_t S_{t-1}^{-1/2})\vert{}\vert{}_F$$

The Shift Ratio ($\kappa_t$): To differentiate between normal evidence accumulation and an abrupt pivot, GDT calculates the ratio of the current step to the historical moving average:


$$\kappa_t = \frac{d_g(S_{t-1}, S_t)}{\frac{1}{k}\sum_{i=1}^{k}d_g(S_{t-i-1}, S_{t-i})}$$

Interference Gate Activation: If $\kappa_t > \kappa_0$ (where $\kappa_0$ is a tunable threshold), the system identifies a "contextual breach." The interference gate triggers, freezing the stale context to prevent it from polluting the new diagnostic trajectory.

The SDT Spectral Trigger Evolution

While the geodesic shift ratio $\kappa_t$ is highly effective for discrete semantic embeddings, it becomes computationally unstable when applied to continuous, high-frequency physiological nodes. SDT resolves this by mapping the pivot detection to the eigenvalues of the evolving hypergraph $G_t$.

Laplacian Monitoring:
At each time step $t$, compute the normalized graph Laplacian matrix for the active subgraph $G_t$:


$$L_t = I - D_t^{-1/2} A_t D_t^{-1/2}$$


where $A_t$ is the adjacency matrix and $D_t$ is the degree matrix.

Fiedler Value Tracking ($\lambda_2$):
Monitor the spectrum of $L_t$, specifically the second smallest eigenvalue, known as the Fiedler value $\lambda_2(L_t)$.

Diagnostic Cohesion: $\lambda_2$ strictly quantifies the algebraic connectivity of the diagnostic hypothesis. Steady growth in $\lambda_2$ represents the clinician gathering corroborating evidence (nodes are densely connected, uncertainty drops).

Spectral Interference Gating:
A sudden, precipitous drop in $\lambda_2$ mathematically indicates that the active subgraph is fracturing into distinct clusters—representing a clinician maintaining stale nodes (the old hypothesis) while simultaneously querying entirely new nodes (the new hypothesis).

New Trigger: The interference gate now triggers when the spectral gap expands rapidly:


$$\Delta \lambda_2 > \tau_{pivot}$$

This functional replacement of $\kappa_t > \kappa_0$ allows the system to detect cognitive pivots across multi-modal data structures in near real-time, purely through graph topology.