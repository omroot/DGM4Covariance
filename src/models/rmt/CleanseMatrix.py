
from typing import Union
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import KFold

from src.models.rmt.MarcenkoPastur import MarcenkoPastur
from src.models.rmt.utils import covariance2correlation
from src.models.rmt.utils import compute_pca


class CleanseMatrix:

    def __init__(self,
                 use_shrinkage: bool,
                 shrinkage_regularizer: float,
                 detone: bool  ,
                 market_components_max_index: int,
                 grid_size: int = 1000,
                 kernel: str = 'gaussian',
                 cv: Union[int, BaseCrossValidator] = 100,
                 kernel_bandwidth: float = 0.01,
                 min_bandwidth_grid_exponent: int = -4,
                 max_bandwidth_grid_exponent: int = 1,
                 bandwidth_grid_size: int = 1000,
                 initial_variance: float = 0.5,
                 epsilon: float = 1e-5,
                 min_q: float = 0.5,
                 max_q: float = 20,
                 q_grid_size: int = 20,
                 verbose: bool = True):
        self.use_shrinkage = use_shrinkage
        self.shrinkage_regularizer =shrinkage_regularizer
        self.detone = detone
        self.market_components_max_index = market_components_max_index
        self.grid_size =grid_size
        self.kernel = kernel
        self.cv = cv
        self.kernel_bandwidth = kernel_bandwidth
        self.min_bandwidth_grid_exponent = min_bandwidth_grid_exponent
        self.max_bandwidth_grid_exponent = max_bandwidth_grid_exponent
        self.bandwidth_grid_size = bandwidth_grid_size
        self.initial_variance = initial_variance
        self.epsilon = epsilon
        self.min_q = min_q
        self.max_q = max_q
        self.q_grid_size = q_grid_size
        self.verbose = verbose

        self.q = None
        self.max_random_eigenvalue = None
        self.variance = None
        self.implied_variance = None
        self.estimated_number_signal_factors = None
        self.cleansed_matrix = None

    def denoise_correlation_matrix(self,
                                   eigenvalues: np.ndarray,
                                   eigenvectors: np.ndarray) -> np.ndarray:
        """Produce a denoised correlation matrix from the set of eigenvalues and eigenvectors and the number of
        signal factors (that can be learned from the Marcenko-Pastur distribution
        Args:
            eigenvalues: the eigen values of the  correlation matrix to be denoised
            eigenvectors: the eigen vectors of the  correlation matrix to be denoised
            number_signal_factors: the number of factors with signal
        Returns:
            correlation_matrix: denoised correlation matrix

        """
        eigenvalues_corrected = np.diag( eigenvalues  ).copy()
        scaler = float(eigenvalues_corrected.shape[0] - self.estimated_number_signal_factors)
        eigenvalues_corrected[self.estimated_number_signal_factors:] = eigenvalues_corrected[self.estimated_number_signal_factors:].sum() / scaler
        eigenvalues_corrected = np.diag(eigenvalues_corrected)
        denoised_correlation_matrix = np.dot(eigenvectors, eigenvalues_corrected).dot(eigenvectors.T)
        scaled_denoised_correlation_matrix = covariance2correlation(denoised_correlation_matrix)
        return scaled_denoised_correlation_matrix

    def shrinkage_denoise_correlation_matrix(self,
                                             eigenvalues: np.ndarray,
                                             eigenvectors: np.ndarray ):
        """ Denoise the correlation matrix through targeted shrinkage
        Args:
            eigenvalues:  the eigen values of the  correlation matrix to be denoised
            eigenvectors: the eigen vectors of the  correlation matrix to be denoised
            number_signal_factors: the number of factors with signal
            alpha: regularization parameter (zero for total shrinkage)

        Returns:
            denoised_correlation_matrix: denoised correlation matrix

        """

        signal_eigenvalues = eigenvalues[:self.estimated_number_signal_factors, :self.estimated_number_signal_factors]
        signal_eigenvectors = eigenvectors[:, :self.estimated_number_signal_factors]

        noise_eigenvalues = eigenvalues[self.estimated_number_signal_factors:, self.estimated_number_signal_factors:]
        noise_eigenvectors = eigenvectors[:, self.estimated_number_signal_factors:]

        signal_correlation_matrix = np.dot(signal_eigenvectors, signal_eigenvalues).dot(signal_eigenvectors.T)

        random_correlation_matrix = np.dot(noise_eigenvectors, noise_eigenvalues).dot(noise_eigenvectors.T)

        denoised_correlation_matrix = signal_correlation_matrix + self.shrinkage_regularizer * random_correlation_matrix + (
                    1 - self.shrinkage_regularizer) * np.diag(np.diag(random_correlation_matrix))

        return denoised_correlation_matrix

    def detone_correlation_matrix(self,
                                  correlation_matrix: np.ndarray,
                                  eigenvalues: np.ndarray,
                                  eigenvectors: np.ndarray):
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
        eigenvalues_market = eigenvalues[:self.market_components_max_index, :self.market_components_max_index]
        eigenvectors_market = eigenvectors[:, :self.market_components_max_index]

        # Calculating the market component correlation
        correlation_market_component = np.dot(eigenvectors_market, eigenvalues_market).dot(eigenvectors_market.T)

        # Removing the market component from the de-noised correlation matrix
        unscaled_detoned_correlation_matrix = correlation_matrix - correlation_market_component

        # Rescaling the correlation matrix to have 1s on the main diagonal
        detoned_correlation_matrix = covariance2correlation(unscaled_detoned_correlation_matrix)

        return detoned_correlation_matrix


    def fit(self, input_matrix: np.ndarray):
        eigenvalues,eigenvectors = compute_pca(input_matrix)
        marcenko_pastur = MarcenkoPastur(grid_size = self.grid_size,
                                         kernel= self.kernel,
                                         cv= self.cv,
                                         kernel_bandwidth=self.kernel_bandwidth,
                                         min_bandwidth_grid_exponent= self.min_bandwidth_grid_exponent,
                                         max_bandwidth_grid_exponent= self.max_bandwidth_grid_exponent,
                                         bandwidth_grid_size= self.bandwidth_grid_size,
                                         initial_variance= self.initial_variance,
                                         epsilon= self.epsilon,
                                         min_q= self.min_q,
                                         max_q= self.max_q,
                                         q_grid_size= self.q_grid_size,
                                         verbose= self.verbose )

        marcenko_pastur.fit(np.diag(eigenvalues))
        self.q = marcenko_pastur.q
        self.max_random_eigenvalue = marcenko_pastur.max_random_eigenvalue
        self.variance = marcenko_pastur.variance
        self.implied_variance = marcenko_pastur.implied_variance
        self.estimated_number_signal_factors = marcenko_pastur.estimated_number_signal_factors

        if self.use_shrinkage:
            correlation_matrix = self.shrinkage_denoise_correlation_matrix(eigenvalues, eigenvectors)
            self.cleansed_matrix = covariance2correlation(correlation_matrix)
        else:
            # correlation_matrix = self.denoise_correlation_matrix(eigenvalues, eigenvectors)
            self.cleansed_matrix =  self.denoise_correlation_matrix(eigenvalues, eigenvectors)
        if self.detone:

            eigenvalues, eigenvectors = compute_pca(self.cleansed_matrix)
            self.cleansed_matrix = self.detone_correlation_matrix( correlation_matrix=self.cleansed_matrix,
                                                                    eigenvalues = eigenvalues,
                                                                    eigenvectors = eigenvectors)



