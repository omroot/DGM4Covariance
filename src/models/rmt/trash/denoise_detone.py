
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import learning_curve ,GridSearchCV
from sklearn.model_selection import LeaveOneOut



def MarcenkoPasturPDF(variance: float ,
                      q : float,
                      pts: int )-> pd.Series():
    """ Compute the probability density function of the Marcenko-Pastur distribution

    Args:
        variance: variance parameter of the Marcenko-Pastur distribution
        q: the ratio T/N where T is the number of rows and N is the number of columns of the data matrix
        In Asset management, the data matrix is
        pts: the number of the data points in the grid the pdf is evaluated on

    Returns:
        pdf: the probability density function of the Marcenko-Pastur distribution with parameters
        variance and q on pts points

    """

    min_eigenvalue = variance *(1 - (1. /q )**.5 )** 2
    max_eigenvalue = variance * (1 + (1. / q) ** .5) ** 2

    eigenvalues_grid = np.linspace(min_eigenvalue, max_eigenvalue, pts)

    pdf = q / (2 * np.pi * variance * eigenvalues_grid) * (
                (max_eigenvalue - eigenvalues_grid) * (eigenvalues_grid - min_eigenvalue)) ** .5

    pdf = pd.Series(pdf, index=eigenvalues_grid)

    return pdf


def getPCA(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the principal components of an input matrix.
    Args:
        matrix: input matrix, typically a covariance or correlation matrix, of dimension NxN

    returns:
        eigenvalues: the eigenvalues , or the variances of the learned principal components,  it is an NxN matrix and
         the eigenvalues are on the diagonal
        eigenvectors: NxN matrix of the eigen vectors, or principal components, each column [:,i] is a normalized
        principal component corresponding to the eigenvalue [i,i]

    """
    # Get eigenvalues,eigenvectors from a Hermitian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    indices = eigenvalues.argsort()[::-1]  # arguments for sorting eVal desc
    eigenvalues = eigenvalues[indices]
    eigenvalues = np.diagflat(eigenvalues)

    eigenvectors = eigenvectors[:, indices]
    return eigenvalues, eigenvectors


def FitEvaluateKernelDensityEstimator(observations: np.ndarray,
                                      kernel_bandwidth: float = .25,
                                      kernel: str = 'gaussian',
                                      evaluation_observations=None) -> pd.Series:
    """Fits a kernel density estimator to an array of input observations
    Args:
        observations: input data
        kernel_bandwidth: the bandwidth of the kernel
        kernel: the kernel to be used
        evaluation_observations: optional array of observation the fitted pdf will be evaluated on, input training
        data will be used instead
    return:
        pdf: probability density function evaluated on the either the training observations or the supplied
        evaluation observations
    """
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(observations.shape) == 1:
        observations = observations.reshape(-1, 1)
    KernelDensityEstimator = KernelDensity(kernel=kernel, bandwidth=kernel_bandwidth).fit(observations)

    if evaluation_observations is None:
        evaluation_observations = np.unique(observations).reshape(-1, 1)
    if len(evaluation_observations.shape) == 1:
        evaluation_observations = evaluation_observations.reshape(-1, 1)
    logProb = KernelDensityEstimator.score_samples(evaluation_observations)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=evaluation_observations.flatten())

    return pdf


def find_optimal_kernel_bandwidth(eigenvalues: np.ndarray) -> dict:
    """This finds the best kernel bandwidth value that fits the best the Kernel Density Estimator
    Args:
        eigenvalues: the eigenvalues the Kernel Density Estimator is fitted on
    Return:
        best_params: a dictionary with the best parameters
    """

    bandwidths = 10 ** np.linspace(-2, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(eigenvalues[:, None])

    return grid.best_params_


def generate_random_covariance(number_total_factors: int,
                               number_signal_factors: int) -> np.ndarray:
    """Generates a random covariance matrix with non random/signal coming from a predefined fixed number of
    factors
    Args:
        number_total_factors:
        number_signal_factors:
    Return:
        covariance_matrix: the generated covariance matrix

    """
    w = np.random.normal(size=(number_total_factors, number_signal_factors))
    covariance_matrix = np.dot(w, w.T)  # random cov matrix, however not full rank
    covariance_matrix += np.diag(np.random.uniform(size=number_total_factors))  # full rank cov
    return covariance_matrix


def covariance2correlation(covariance_matrix: np.ndarray) -> np.ndarray:
    """Converts a covariance matrix into a correlation matrix
    Args:
        covariance_matrix: input covariance matrix
    Returns:
        correlation_matrix: output correlation matrix
    """
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std, std)
    correlation_matrix[correlation_matrix < -1] = -1  # numerical error
    correlation_matrix[correlation_matrix > 1] = 1  # numerical error
    return correlation_matrix


def empirical_vs_theoretical_pdf_error(variance: np.ndarray,
                                       eigenvalues: np.ndarray,
                                       q: float,
                                       kernel_bandwidth: float,
                                       pts: int = 1000) -> float:
    """ Computes the sum of squared errors between the theoretical MarcenKo-Pastur distribution
    and the empirical distribution computed using Kernel Density Estimation
    Args:
        variance: parameter for the MarcenKo-Pastur distribution
        eigenvalues: input observation
        q: MP distribution parameter
        kernel_bandwidth: kernel bandwith for the Kernel Density Estimator
        pts: number of data points on the grid for the theoretical distribution pdf
    return:
        sse: sum of squared errors between theoretical and empirical distribution
    """

    var = variance[0]
    theoretical_pdf = MarcenkoPasturPDF(var, q, pts)  # theoretical pdf

    empirical_pdf = FitEvaluateKernelDensityEstimator(observations=eigenvalues,
                                                      kernel_bandwidth=kernel_bandwidth,
                                                      evaluation_observations=theoretical_pdf.index.values)  # empirical pdf
    sse = np.sum((theoretical_pdf - empirical_pdf) ** 2)
    return sse


def find_max_random_eigenvalue(eigenvalues,
                               q,
                               kernel_bandwidth):
    """

    :param eigenvalues:
    :param q:
    :param kernel_bandwidth:
    Returns:
        max_random_eigenvalue:
        variance:

    """
    # Find max random eVal by fitting Marcenko’s distribution
    out = minimize(lambda *x: empirical_vs_theoretical_pdf_error(*x),
                   x0=np.array(.5),
                   args=(eigenvalues, q, kernel_bandwidth),
                   bounds=((1E-5, 1 - 1E-5),))
    print(f"found errPDFs {out['x'][0]}")

    if out['success']:
        variance = out['x'][0]
    else:
        variance = 1
    max_random_eigenvalue = variance * (1 + (1. / q) ** .5) ** 2
    return max_random_eigenvalue, variance


def denoise_correlation_matrix(eigenvalues: np.ndarray,
                               eigenvectors: np.ndarray,
                               number_signal_factors: int) -> np.ndarray:
    """Produce a denoised correlation matrix from the set of eigenvalues and eigenvectors and the number of
    signal factors (that can be learned from the Marcenko-Pastur distribution
    Args:
        eigenvalues: the eigen values of the  correlation matrix to be denoised
        eigenvectors: the eigen vectors of the  correlation matrix to be denoised
        number_signal_factors: the number of factors with signal
    Returns:
        correlation_matrix: denoised correlation matrix

    """
    eigenvalues_corrected = np.diag(eigenvalues).copy()
    eigenvalues_corrected[number_signal_factors:] = eigenvalues_corrected[number_signal_factors:].sum() / float(
        eigenvalues_corrected.shape[0] - number_signal_factors)
    eigenvalues_corrected = np.diag(eigenvalues_corrected)
    correlation_matrix = np.dot(eigenvectors, eigenvalues_corrected).dot(eigenvectors.T)
    correlation_matrix = covariance2correlation(correlation_matrix)
    return correlation_matrix


def shrinkage_denoise_correlation_matrix(eigenvalues,
                                         eigenvectors,
                                         number_signal_factors,
                                         alpha=0):
    """ Denoise the correlation matrix through targeted shrinkage
    Args:
        eigenvalues:  the eigen values of the  correlation matrix to be denoised
        eigenvectors: the eigen vectors of the  correlation matrix to be denoised
        number_signal_factors: the number of factors with signal
        alpha: regularization parameter (zero for total shrinkage)

    Returns:
        denoised_correlation_matrix: denoised correlation matrix

    """

    signal_eigenvalues = eigenvalues[:number_signal_factors, :number_signal_factors]
    signal_eigenvectors = eigenvectors[:, :number_signal_factors]

    random_eigenvalues = eigenvalues[number_signal_factors:, number_signal_factors:]
    random_eigenvectors = eigenvectors[:, number_signal_factors:]

    signal_correlation_matrix = np.dot(signal_eigenvectors, signal_eigenvalues).dot(signal_eigenvectors.T)

    random_correlation_matrix = np.dot(random_eigenvectors, random_eigenvalues).dot(random_eigenvectors.T)

    denoised_correlation_matrix = signal_correlation_matrix + alpha * random_correlation_matrix + (1 - alpha) * np.diag(
        np.diag(random_correlation_matrix))

    return denoised_correlation_matrix


def detone_correlation_matrix(correlation_matrix,
                              eigenvalues,
                              eigenvectors,
                              market_components_max_index=1):
    """Detones the denoised correlation matrix from its eigen values and vectors by removing the market component.
    Args:
        correlation_matrix: input correlation matrix
        eigenvalues:the eigen values of the  correlation matrix to be detoned
        eigenvectors:the eigen vectors of the  correlation matrix to be detoned
        market_components_max_index: maximum index of the market component (they could be more than one)
    Returns:
        detoned_correlation_matrix: detoned correlation matrix

    """

    # Getting the eigenvalues and eigenvectors related to market component
    eigenvalues_market = eigenvalues[:market_components_max_index, :market_components_max_index]
    eigenvectors_market = eigenvectors[:, :market_components_max_index]

    # Calculating the market component correlation
    correlation_market_component = np.dot(eigenvectors_market, eigenvalues_market).dot(eigenvectors_market.T)

    # Removing the market component from the de-noised correlation matrix
    unscaled_detoned_correlation_matrix = correlation_matrix - correlation_market_component

    # Rescaling the correlation matrix to have 1s on the main diagonal
    detoned_correlation_matrix = covariance2correlation(unscaled_detoned_correlation_matrix)

    return detoned_correlation_matrix


