1. Hypergraph Construction & Node Initialization
    To support encoding medical documents and forming semantic/temporal edges:
        "ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission": Foundational for dense clinical semantic embeddings.  
        "Learning with Hypergraphs: Clustering, Classification, and Embedding": The foundational framework for constructing hypergraphs and defining normalized hypergraph Laplacian matrices.
        "Session-based Recommendation with Graph Neural Networks": Validates the use of graph structures to model dynamic, session-based user sequences. 

2. Spectral Pivot Detection
    To support your use of the Fiedler value ($\lambda_2$) as a replacement for the Riemannian metric:
        "Algebraic Connectivity of Graphs": The original, foundational paper establishing the second smallest eigenvalue of the Laplacian (Fiedler value) as the primary measure of structural cohesion.
        "Geodesic Diagnostic Trajectories (GDT): A Differential Geometry Framework for Adaptive Decision Support and Operational Optimization in EHR Systems" (main.pdf): You must cite this to contextualize your Fiedler drop as a functional replacement for the continuous Riemannian geodesic shift ratio $\kappa_t$ used to detect abrupt topic pivots. 

3. Graph-Signal Context Attenuation
    To support severing pre-pivot edges and applying high-pass filtering:
        "The Emerging Field of Signal Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and Other Irregular Domains": The definitive primer on spectral graph filtering and defining high-pass/low-pass filters on graph signals.
        "Geodesic Diagnostic Trajectories (GDT)...": Cite this to benchmark your subgraph decoupling against their curvature-gated interference mechanism, which provably attenuates stale context during non-linear chart review.  

4. Anticipatory Prefetch Engine
    To support the biased random walk and optimal control framing:
        "Geodesic Diagnostic Trajectories (GDT)...": This paper explicitly establishes the anticipatory prefetch engine as an optimal control problem designed to minimize expected clinical decision delay subject to cognitive-memory constraints.  
        "The PageRank Citation Ranking: Bringing Order to the Web": Foundational for transition-probability-biased random walks on graphs