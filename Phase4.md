Phase 4: Anticipatory Prefetch Engine & Simulation Interface

Objective: Evolve the retrieval engine into a predictive world model driven by optimal control constraints.

Optimal Control Formulation:

Similar to GDT, prefetching is framed mathematically to minimize expected clinical decision delay, operating natively on graph topologies. Let $u_t$ denote the control action (the subset of nodes selected for prefetching from the global graph).

The system optimizes a continuous loss function $J(u_t)$ that balances temporal retrieval delay, structural relevance, and cognitive load bounds:

$$J(u_t) = \mathbb{E}\left[ \alpha \mathcal{L}_{delay}(G_{t+1}, u_t) - \beta \mathcal{R}_{struct}(u_t) + \gamma \Omega(u_t) \right]$$

Where $\mathcal{L}_{delay}$ calculates the time saved by having nodes readily available, $\mathcal{R}_{struct}$ rewards structural similarity to the active hypothesis, and $\Omega(u_t)$ penalizes overwhelming the clinician's cognitive bandwidth.

Biased Traversal (Spectrally-Biased Random Walk):

Replace the JEPA-based target network with a parameterized random walk on the global EHR graph $G_{global}$.

Let standard transition probabilities be $P_0(v_j \vert{} v_i) = \frac{A_{ij}}{D_{ii}}$. The system instead computes a spectrally-biased transition probability, mapping nodes into a spectral embedding space $\phi(v)$ to heavily weight trajectories structurally similar to the active subgraph:

$$P(v_j \vert{} v_i) = \frac{P_0(v_j \vert{} v_i) \exp(-\tau \vert{}\vert{}\phi(v_i) - \phi(v_j)\vert{}\vert{}_2^2)}{\sum_{k \in \mathcal{N}(i)} P_0(v_k \vert{} v_i) \exp(-\tau \vert{}\vert{}\phi(v_i) - \phi(v_k)\vert{}\vert{}_2^2)}$$

This spectral biasing mechanism allows the system to simulate branching physiological futures and natively forecast the next clinical intent across multi-modal nodes.

Cognitive Gating:

To adopt the cognitive safety principles of GDT (mitigating premature closure and automation bias), the prefetch volume is mathematically restricted.

Let $H(v \vert{} G_t)$ represent the cognitive load increment of presenting node $v$ relative to the current active context. The optimal control action $u_t$ is strictly subjected to the human cognitive-memory constraint (informed by Sweller's cognitive-load theory):

$$\sum_{v \in u_t} H(v \vert{} G_t) \leq C_{max}$$

This directly prevents UI overwhelming and ensures the human-in-the-loop retains full agency over the diagnostic trajectory.

