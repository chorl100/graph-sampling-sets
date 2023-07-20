import numpy as np
from scipy import sparse


def gft(U, s):
    """
    Computes the graph Fourier transform (GFT).
    Transforms a graph signal from the graph to the frequency domain.
    The GFT is defined as
        $\hat s = U^Ts.
    :param U: Graph Fourier basis (eigenvectors of the graph Laplacian) of shape (n, k)
    :param s: signal vector of size n (graph domain)
    :return: transformed signal in the frequency domain
    """
    return U.T @ s.reshape(-1, 1)


def igft(U, s_hat):
    """
    Computes the inverse graph Fourier transform (IGFT).
    Transforms a signal from the frequency to the graph domain.
    The iGFT is defined as
        $s = U\hat s$.
    :param U: Graph Fourier basis (eigenvectors of the graph Laplacian) of shape (n, k)
    :param s_hat: signal vector of size n (frequency domain)
    :return: transformed signal in the graph domain
    """
    return U @ s_hat.reshape(-1, 1)


def eigh_sparse(L: sparse.csr_matrix):
    eigvals, eigvecs = np.linalg.eigh(np.array(L.todense().astype('float')))
    return eigvals, eigvecs
