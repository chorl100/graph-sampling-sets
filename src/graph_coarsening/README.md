# graph-coarsening package

Copied from https://github.com/loukasa/graph-coarsening.

Multilevel graph coarsening algorithm with spectral and cut guarantees.

The code accompanies the paper [Graph reduction with spectral and cut guarantees](https://www.jmlr.org/papers/volume20/18-680/18-680.pdf) by Andreas Loukas published at JMLR/2019.

In addition to the introduced [**variation**](https://www.jmlr.org/papers/volume20/18-680/18-680.pdf) methods, the code provides implementations of [**heavy-edge matching**](https://proceedings.mlr.press/v80/loukas18a.html), [**algebraic distance**](https://epubs.siam.org/doi/abs/10.1137/100791142?casa_token=tReVSPG0pBIAAAAA:P3BxPcyiSNkuxP5mOz8s9I7CN1tFQaMUTjyVHvb7PphqsGDy91ybcmAmECTYOeN2l-ErcpXuuA), [**affinity**](https://epubs.siam.org/doi/abs/10.1137/110843563?mobileUi=0), and [**Kron reduction**](https://motion.me.ucsb.edu/pdf/2011d-db.pdf) (adapted from [pygsp](https://pygsp.readthedocs.io/en/stable)).   

## Paper abstract 
Can one reduce the size of a graph without significantly altering its basic properties? 
The graph reduction problem is hereby approached from the perspective of restricted spectral approximation, a modification of the spectral similarity measure used for graph sparsification. This choice is motivated by the observation that restricted approximation carries strong spectral and cut guarantees, and that it implies approximation results for unsupervised learning problems relying on spectral embeddings. The article then focuses on coarsening - the most common type of graph reduction. Sufficient conditions are derived for a small graph to approximate a larger one in the sense of restricted approximation. These findings give rise to algorithms that, compared to both standard and advanced graph reduction methods, find coarse graphs of improved quality, often by a large margin, without sacrificing speed.
