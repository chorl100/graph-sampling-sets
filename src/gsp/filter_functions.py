import numpy as np
import numpy.linalg as la
from scipy import signal

from src.gsp import fourier


def apply_filter(L: np.ndarray, s: np.ndarray, filter_func):
    """
    Applies a filter function to a graph signal. The filter is applied in the frequency domain.
    :param L: graph Laplacian
    :param s: signal vector
    :param filter_func: filter function
    :return: filtered signal
    """
    # compute Fourier basis U
    lamda, U = la.eigh(L)
    # transform signal to Fourier domain
    s_hat = fourier.gft(U, s)
    # apply filter
    s_hat = np.diag(filter_func(lamda)) @ s_hat
    # transform signal back to graph domain
    s_filtered = fourier.igft(U, s_hat)
    return np.asarray(s_filtered)


def low_pass_filter(lamda: np.ndarray, lamda_upper: float):
    """
    Passes low frequent signals (corresponding to small eigenvalues)
    while attenuating higher frequencies corresponding to eigenvalues larger than lamda_upper.
    :param lamda: eigenvalues
    :param lamda_upper: upper bound cut-off frequency
    :return: filtered frequencies
    """
    return lamda <= lamda_upper


def high_pass_filter(lamda: np.ndarray, lamda_lower: float):
    """
    Passes high frequent signals (corresponding to high eigenvalues)
    while attenuating lower frequencies corresponding to eigenvalues smaller than lamda_lower.
    :param lamda: eigenvalues
    :param lamda_lower: lower bound cut-off frequency
    :return: filtered frequencies
    """
    return lamda >= lamda_lower


def band_pass_filter(lamda: np.ndarray, lamda_lower: float, lamda_upper: float):
    return (lamda >= lamda_lower) * (lamda <= lamda_upper)


def butter_low_pass_filter(lamda: np.ndarray, lamda_upper: float, order=2):
    """
    Smooth low-pass filter.
    :param lamda: eigenvalues
    :param lamda_upper: cutoff eigenvalue (larger values are attenuated)
    :param order: filter order
    :return: filtered eigenvalues
    """
    fs = 2.1 * max(lamda)
    nyq = 0.5 * fs
    normal_cutoff = lamda_upper / nyq
    # filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    # frequency response
    w, h = signal.freqz(b, a, worN=len(lamda))
    return abs(h)


def butter_high_pass_filter(lamda: np.ndarray, lamda_lower: float, order=2):
    """
    Smooth high-pass filter.
    :param lamda: eigenvalues
    :param lamda_lower: cutoff eigenvalue (smaller values are attenuated)
    :param order: filter order
    :return: filtered eigenvalues
    """
    fs = 2.1 * max(lamda)
    nyq = 0.5 * fs
    normal_cutoff = lamda_lower / nyq
    # filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    # frequency response
    w, h = signal.freqz(b, a, worN=len(lamda))
    return abs(h)


def constant_filter(lamda, c):
    return c * np.eye(lamda.shape[0]) * lamda


def heat_filter(lamda, c):
    return np.exp(-c * lamda)
