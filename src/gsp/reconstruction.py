import numpy as np
from scipy import sparse
# necessary import because sparse.linalg is not recognized (possible reason old scipy version)
import scipy.sparse.linalg


def reconstruct_signal(L: sparse.csr_matrix, sampling_set: list, s_sampled: np.ndarray, mu: float = 0.01):
    """
    Reconstructs a signal of size n from a sampled signal of size k<n using
    Graph Laplacian Regularization (GLR).
    :param L: laplacian matrix of shape (n, n)
    :param sampling_set: list of sampled nodes
    :param s_sampled: sampled signal of size k
    :param mu: trade-off parameter between reconstruction error and signal smoothness (regularization)
    :return: reconstructed signal of size n
    """
    n_nodes = L.shape[0]
    a = np.zeros(n_nodes, dtype=np.int8)
    a[sampling_set] = 1
    A = sparse.diags(a, format="csr")
    k = len(sampling_set)
    H = sparse.lil_matrix((k, n_nodes), dtype=np.int8)
    H[range(k), sampling_set] = 1
    Ax = A + mu*L
    b = H.T @ s_sampled.reshape(-1, 1)
    b = sparse.csc_matrix(b)
    return sparse.linalg.spsolve(Ax, b)


def reconstruct_signal_direct(L: sparse.csr_matrix, sampling_set: list, s: np.ndarray, mu: float = 0.01):
    """
    Reconstructs a signal of size n from a sampled signal of size k<n using
    Graph Laplacian Regularization (GLR).
    Direct computation is _discouraged_ because it involves computing the inverse of the reconstruction matrix
    which is costly and _unstable_.
    :param L: graph laplacian matrix with shape (n, n)
    :param sampling_set: list of sampled nodes
    :param s: sampled signal vector of size k
    :param mu: trade-off parameter between reconstruction error and signal smoothness
    :return: reconstructed signal of size n
    """
    reconstr_mat = reconstruction_matrix(L, sampling_set, mu)
    n_nodes = L.shape[0]
    k = len(sampling_set)
    H = sparse.lil_matrix((k, n_nodes), dtype=np.int8)
    H[range(k), sampling_set] = 1
    H = sparse.csr_matrix(H)
    s_reconst = sparse.linalg.inv(reconstr_mat) @ H.T @ s.reshape(-1, 1)
    return s_reconst.flatten()


def reconstruction_matrix(L: sparse.csr_matrix, sampling_set: list, mu: float = 0.01):
    """
    Computes the reconstruction matrix B (as in the paper).
    Can be used for direct calculation of the reconstructed signal.
    :param L: graph laplacian matrix
    :param sampling_set: list of sampled nodes
    :param mu: trade-off parameter between reconstruction error and signal smoothness
    :return: reconstruction matrix
    """
    n_nodes = L.shape[0]
    a = np.zeros(n_nodes, dtype=np.int8)
    a[list(sampling_set)] = 1
    B = sparse.diags(a, format="csr") + mu * L
    B = sparse.csc_matrix(B)
    return B


def mse(a: np.ndarray, b: np.ndarray):
    """
    Computes the mean squared error (MSE) between two arrays.
    :param a: vector
    :param b: vector
    :return: MSE between arrays a and b
    """
    return np.mean(np.square(a-b))


def rmse(a: np.ndarray, b: np.ndarray):
    """
    Computes the root mean squared error (RMSE) between two arrays.
    :param a: vector
    :param b: vector
    :return: RMSE between arrays a and b
    """
    return np.sqrt(mse(a, b))


def norm_error_norm(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the normalized error norm between two vectors a and b, normalized wrt. vector a.
    :param a: vector
    :param b: vector
    :return: normalized error norm
    """
    return np.linalg.norm(a-b) / np.linalg.norm(a)


def snr(s_true: np.ndarray, s_rec: np.ndarray) -> float:
    """
    Computes the signal-to-noise ratio (SNR) between a reconstructed signal s_rec and ground
    truth signal s_true.
    :param s_true: ground truth signal
    :param s_rec: reconstructed/predicted signal
    :return: SNR between s_true and s_rec
    """
    return -20 * np.log10(norm_error_norm(s_true, s_rec))
