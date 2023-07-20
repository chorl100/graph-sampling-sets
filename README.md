# Fast Graph Sampling Set Selection Using Gershgorin Disc Alignment

This repository migrates the code of the paper "Fast Graph Sampling Set Selection Using
Gershgorin Disc Alignment" by Bai et al. [[1]](#1) from Matplotlib/C++ to Python.

Moreover, it adds functionalities for customer analytics, among them deep detractor classification.

Quick jump to commands for [Workflows](#workflows).

A demo notebook for signal reconstruction can be found [here](notebooks/use_cases/graph_sampling_demo.ipynb).

## Requirements

### Console

The code was developed and tested on Python 3.8.  
To access the data, you need to be on the GCP.
[Here](cluster_class.yaml) you can find a useful configuration for your Dataproc cluster.
You can install all necessary dependencies with the command

```shell
$ pip install -r requirements.txt
```

Note: If you encounter the error `AttributeError: module 'faiss' has no attribute 'StandardGpuResources'`, 
then something went wrong during the installation of faiss-gpu. In this case run

```shell
$ pip install --force-reinstall faiss-gpu
```

to fix the problem.

### Jupyter

When working in a Jupyter notebook on the GCP, you can install the requirements from within a code cell with the command

```shell
$ !pip install -r requirements.txt
```

Note the "!" before the pip call.


## Working on GCP

When running code from the console, remember to **export your Python path** with

```shell
$ export PYTHONPATH=$PWD
```
to export your current directory as working directory. Otherwise, you might get a `ModuleNotFoundError`.


## Project Tree

````shell
src
│   __init__.py
│
├───bioinfo
│   │   centrality_analytics.py
│   │   pdb_centrality.py
│   │   utils.py
├───db
│   │   big_query.py
│   │   constants.py
│   │   nps.py
│   │   preprocessing.py
│   │   zip_code_mapper.py
│   │   __init__.py
│   │
│   ├───resources
│   │       imputation_dictionary.csv
├───eval
│   |   eval_reconstruction.py
│   |   __init__.py
├───gershgorin
│   │   bs_gda.py
│   │   bucket_queue.py
│   │   disc_alignment.py
│   │   greedy_sampling.py
│   │   solving_set_covering.py
│   │   __init__.py
├───graph
│   │   gauss_similarity_graph.py
│   │   graph.py
│   │   graph_builder.py
│   │   graph_tools.py
│   │   metrics.py
│   │   nearest_neighbors.py
│   │   nearest_neighbors_gpu.py
│   │   neighborhood_regression.py
│   │   sample_evaluation.py
│   │   similarity_graph.py
│   │   __init__.py
├───graph_coarsening
│       coarsening_utils.py
│       graph_lib.py
│       graph_utils.py
│       maxWeightMatching.py
│       README.md
│       version.py
│       __init__.py
├───gsp
│   │   filter_functions.py
│   │   fourier.py
│   │   laplace_utils.py
│   │   metrics.py
│   │   reconstruction.py
│   │   signal.py
│   │   __init__.py
├───metrics
│       metrics.py
│       __init__.py
├───utils
│   │   colors.py
│   │   data_handler.py
│   │   layout.py
│   │   plotting.py
│   │   yaml_reader.py
│   │   __init__.py
````


## Usage

### Code example

To run the sampling set selection method on a given graph, 
you can run the following code in Python:

````python
# only needed for creation of demo graph and adj. matrix
import networkx as nx
# core packages
from src.gershgorin.bs_gda import bs_gda
from src.graph.graph import Graph

# create a graph (here Erdős-Rényi graph) or use an existing one
n_nodes = 500
G = nx.fast_gnp_random_graph(n_nodes, p=0.5)
sampling_budget = 100  # max. number of sampled nodes
graph = Graph(nx.adjacency_matrix(G))  # wrapper graph class
# select sampling set
sampling_set, _ = bs_gda(graph, sampling_budget, parallel=False)
print("Number of sampled nodes:", len(sampling_set))
````

For examples of nearest neighbor graph construction, signal reconstruction, and more, 
please refer to the [notebooks](notebooks).

### Workflows

The workflows in [bin](bin/customer_analytics) perform tasks typically needed for customer analytics.
Each workflow can be configured with its corresponding configuration file.
The commands follow a certain structure:
```shell
$ python bin/customer_analytics/{workflow}.py -c config/customer_analytics/{workflow}.yml
```

Please refer to the configuration files to see which input files are required and which outputs are generated.

Usually the first steps to take are preprocessing, graph building and signal generation.

#### Preprocessing
Loads and preprocesses the customer data for one month. Only customers who participated at the NPS survey are considered.
```shell
python bin/customer_analytics/data_preprocessing.py -c config/customer_analytics/data_preprocessing.yml
```

#### Graph building
Constructs a similarity graph based on customer data. Possible graphs are ["knn", "knn_gpu", "gauss"].
```shell
python bin/customer_analytics/graph_builder.py -c config/customer_analytics/graph_builder.yml
```

#### Set selection
Selects a sampling set of given size.
```shell
python bin/customer_analytics/sampling_set_selection.py -c config/customer_analytics/sampling_set_selection.yml
```

#### Set extension
Extends a given sampling set (preselection).
```shell
python bin/customer_analytics/sampling_set_extension.py -c config/customer_analytics/sampling_set_extension.yml
```

#### Evaluate selected samples
Compares two sampling sets, regarding one as ground truth and the other as prediction. 
Calculates some performance statistics.
```shell
python bin/customer_analytics/eval_sampling_selection.py -c config/customer_analytics/eval_sampling_selection.yml
```

#### Network Drawing
Draws a customer graph.
```shell
python bin/customer_analytics/network_drawing.py -c config/customer_analytics/network_drawing.yml
```

#### Signal generation
Generates a signal vector. Possible signals are ["nps", "gs1", "gs2", "bandlimited"].  
GS1 is a bandlimited signal, GS2 is an approximately smooth but not bandlimited signal. 
The option "bandlimited" allows customizable coefficients in the linear combination of the eigenvectors.
```shell
python bin/customer_analytics/signal_generation.py -c config/customer_analytics/signal_generation.yml
```

#### Graph signal drawing
Draws a signal on a graph.
```shell
python bin/customer_analytics/graph_signal_drawing.py -c config/customer_analytics/graph_signal_drawing.yml
```

#### Signal reconstruction
Reconstructs a signal from one or more sampling sets. Then error and smoothness metrics are calculated.
```shell
python bin/customer_analytics/signal_reconstruction.py -c config/customer_analytics/signal_reconstruction.yml
```

#### Signal reconstruction for growing sampling budget
Given a graph and a signal, it measures the reconstruction quality for increasing sample size.
```shell
python bin/customer_analytics/eval_growing_sampling_budget.py -c config/customer_analytics/eval_growing_sampling_budget.yml
```

#### Signal reconstruction for growing p-hop neighborhood
Given a graph and a signal, it measures the reconstruction quality for increasing p-hop neighborhood.
```shell
python bin/customer_analytics/eval_growing_p_hops.py -c config/customer_analytics/eval_growing_p_hops.yml
```

#### Deep Detractor Detection
Reconstructs a signal from a small subsample of the observations using graph sampling. 
Then classifies customers based on the reconstructed signal.
```shell
python bin/customer_analytics/deep_detractor_detection.py -c config/customer_analytics/deep_detractor_detection.yml
```

#### Extra: Download the results
To download all output files at once via Jupyter, the easiest way is to zip the `out` folder with
```shell
$ zip -r out.zip out
```
Then download the .zip-file (here out.zip).


## References

<a id="1">[1]</a> 
Yuanchao Bai, Fen Wang, Gene Cheung, et al. 
“Fast Graph Sampling Set Selection Using Gershgorin Disc Alignment”.  
In: IEEE Transactions on Signal Processing 68 (2020),
pp. 2419–2434. URL: https://doi.org/10.1109%2Ftsp.2020.2981202.
