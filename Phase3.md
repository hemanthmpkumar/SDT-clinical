Phase 3: Graph-Signal Context Attenuation

Objective: Bound context contamination and prevent stale hypotheses from polluting downstream trajectory simulation by executing a mathematically rigorous Fiedler cut.

Spectral Filtering:

Upon trigger activation ($\Delta \lambda_2 > \tau_{pivot}$), the system applies a high-pass graph filter to the node signals. Let $x$ represent the signal of contextual relevance across the nodes. The filtered signal is $\tilde{x} = H(L_t)x$, where the filter $H$ actively suppresses low-frequency structural components (which heavily correlate with the pre-pivot stale context).

Subgraph Decoupling (The Fiedler Cut):

When the spectral gap triggers, the system isolates the structural fracture using the Fiedler vector $v_2$ (the eigenvector associated with the Fiedler value $\lambda_2$).

The entries of $v_2$ assign a continuous value to each node in the active subgraph $G_t$.

Partitioning: The graph is mathematically partitioned into two disjoint sets based on the algebraic sign of the Fiedler vector components:

$$V_{stale} = \{i \in V_t \mid v_2(i) < 0\}$$

$$V_{active} = \{i \in V_t \mid v_2(i) \geq 0\}$$

Severing Edges: All hyperedges connecting $V_{stale}$ (the old diagnostic trajectory) and $V_{active}$ (the newly forming hypothesis) are systematically severed. This enforces a strict, instantaneous bound on information interference, directly advancing GDT's Theorem 3.2 (Curvature-Gated Contamination Decay) from Riemannian geometry into discrete graph space.

Embedding Formulation:

Once the stale subgraph is decoupled, the system aggregates only the nodes in $V_{active}$ to generate a cleanly filtered, highly focused retrieval vector.

This isolated vector serves as the seed for generating the next immediate query state, completely free of latent-context pollution from the prior diagnostic path.