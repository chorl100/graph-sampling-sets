import faiss
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.graph import graph_tools


class NearestNeighborsGPU:
    """
    Class for efficient computation of the k-nearest neighbors.
    Computes both distances and indices to the nearest neighbors.
    Supports both CPU and GPU computations.
    """

    def __init__(self, device="cpu"):
        """
        Initializes the parameter k and the device on which to perform the computations.
        GPU usage accelerates the computations.
        :param device: device type; either "cpu" or "gpu"
        """
        self.index = None
        self.device = device
        self.res = None

    def fit(self, X):
        """
        Fits an index to the feature matrix. The features must have the data type float32 (required by faiss library).
        :param X: feature matrix
        """
        # create an index structure
        if self.device == "gpu":
            if type(X) == pd.DataFrame:
                X = torch.tensor(np.ascontiguousarray(X.to_numpy()), dtype=torch.float32).cuda()
            elif X.dtype != torch.float32:
                X = X.type(torch.float32).cuda()
            # use a single GPU
            res = faiss.StandardGpuResources()
            # build a flat GPU index
            self.index = faiss.GpuIndexFlatL2(res, X.shape[1])
        else:
            if X.to_numpy().dtype != np.float32:
                # cast features to right format
                X = np.ascontiguousarray(X.to_numpy().astype(np.float32))
            # build a flat CPU index
            self.index = faiss.IndexFlatL2(X.shape[1])
        # add features to index
        self.index.add(X)

    def predict(self, X, k: int, include_self=False):
        """
        Computes the k-nearest neighbors distance matrix and neighbor indices for a feature matrix.
        :param X: feature matrix
        :param k: number of neighbors to consider
        :param include_self: whether to mark each sample as the first nearest neighbor to itself
        :return: distance matrix, indices of k-nearest neighbors per node
        """
        if self.device == "gpu":
            if type(X) == pd.DataFrame:
                X = torch.tensor(np.ascontiguousarray(X.to_numpy()), dtype=torch.float32).cuda()
            elif X.dtype != torch.float32:
                X = X.type(torch.float32)
        if not include_self:
            # query k+1 neighbors because the first neighbor (node itself) will be removed
            distances, neighbors_indices = self.index.search(X, k=k + 1)
            # since every entry is the closest to itself, we need to remove the first column
            neighbors_indices = neighbors_indices[:, 1:]
        else:
            distances, neighbors_indices = self.index.search(X, k=k)
        if self.device == "gpu":
            distances, neighbors_indices = distances.cpu(), neighbors_indices.cpu()
        return distances, neighbors_indices

    def knn(self, X, k: int, include_self=False):
        """
        Fits an index to a feature matrix and queries the k-nearest neighbors for each row.
        Same as jointly applying fit and predict.
        :param X: feature matrix
        :param k: number of neighbors to consider
        :param include_self: whether to mark each sample as the first nearest neighbor to itself
        :return: distance matrix, indices of k-nearest neighbors per node
        """
        if self.res is None:
            self.res = faiss.StandardGpuResources()
        if self.device == "gpu":
            if type(X) == pd.DataFrame:
                X = torch.tensor(np.ascontiguousarray(X.to_numpy()), dtype=torch.float32).cuda()
            elif X.dtype != torch.float32:
                X = X.type(torch.float32).cuda()
            distances, indices = faiss.knn_gpu(self.res, X, X, k=k + 1)
            # move data to CPU
            distances, indices = distances.cpu(), indices.cpu()
        else:
            if X.to_numpy().dtype != np.float32:
                X = np.ascontiguousarray(X.to_numpy().astype(np.float32))
            distances, indices = faiss.knn(X, X, k=k + 1)
        if not include_self:
            # drop first column
            indices = indices[:, 1:]
        else:
            # drop last column
            indices = indices[:, :-1]
        return distances, indices

    def knn_multi_gpu(self, X, k: int, include_self=False):
        """
        Fits an index to a feature matrix and queries the k-nearest neighbors for each row.
        Same as jointly applying fit and predict. Distributes the data on all available GPUs.
        :param X: feature matrix
        :param k: number of neighbors to consider
        :param include_self: whether to mark each sample as the first nearest neighbor to itself
        :return: distance matrix, indices of k-nearest neighbors per node
        """
        # create a flat CPU index
        cpu_index = faiss.IndexFlatL2(X.shape[1])
        # map the index to all available GPUs
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        # add the data to the index
        gpu_index.add(X)
        # compute the k-nearest neighbor indices
        distances, neighbor_indices = gpu_index.search(X, k)
        if not include_self:
            # drop first column
            neighbor_indices = neighbor_indices[:, 1:]
        else:
            # drop last column
            neighbor_indices = neighbor_indices[:, :-1]
        return distances, neighbor_indices

    @staticmethod
    def to_adj_matrix(neighbors_idx, distances=None, weighted=False, directed=False) -> sparse.csr_matrix:
        """
        Transforms the index matrix of the k-nearest neighbors to a sparse weight matrix.
        :param neighbors_idx: matrix with indices of the k-nearest neighbors for each node
        :param distances: distances to the k-nearest neighbors of shape (n, k) or (n, k+1)
            (if include_self was true in knn)
        :param weighted: whether to return a weighted or binary adjacency matrix
        :param directed: whether the resulting graph should be directed or undirected
        :return: sparse weight matrix of shape (n, n)
        """
        # build adjacency matrix from neighbors index
        n_nodes = neighbors_idx.shape[0]
        W = sparse.lil_matrix((n_nodes, n_nodes))
        if weighted:
            dist = distances.clone().detach()
            # if distance matrix includes node itself
            if neighbors_idx.shape[1] < dist.shape[1]:
                # remove first column
                dist = dist[:, 1:]
        for i in range(n_nodes):
            W[i, neighbors_idx[i]] = dist[i] if weighted else 1
        W = sparse.csr_matrix(W)
        if directed:
            return W
        return graph_tools.symmetrize_adj(W, weighted)

    @staticmethod
    def save(matrix, filepath):
        """
        Saves the sparse adjacency matrix to a file in '.npz' format.
        :param matrix: sparse adjacency matrix
        :param filepath: path to save file
        """
        assert matrix is not None
        sparse.save_npz(filepath, matrix)
