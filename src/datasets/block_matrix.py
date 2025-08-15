import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from scipy.linalg import block_diag
from typing import Optional


class BlockMatrixGenerator:
    """
    A class used to generate covariance and correlation matrices.

    Methods
    -------
    covariance2correlation(covariance_matrix: np.ndarray) -> np.ndarray:
        Converts a covariance matrix into a correlation matrix.

    generate_sub_covariance(number_observations: int, number_factors: int, sigma: float, random_state: Optional[int] = None) -> np.ndarray:
        Generates a sub-covariance matrix from highly correlated factors.

    generate_random_block_covariance(number_factors: int, number_blocks: int, minimum_block_size: int = 1, sigma: float = 1.0, random_state: Optional[int] = None) -> np.ndarray:
        Generates a random block covariance matrix.

    generate_random_block_correlation_matrix(number_factors: int, number_blocks: int, minimum_block_size: int = 1, random_state: Optional[int] = None) -> pd.DataFrame:
        Generates a random block correlation matrix.
    """

    def __init__(self,
                 number_factors: int,
                 number_blocks: int,
                 minimum_block_size: int = 1,
                 sigma_signal: float = 1.0,
                 sigma_noise: float = 1.0,
                 random_state: int = None
                 ):
        self.number_factors = number_factors
        self.number_blocks = number_blocks
        self.minimum_block_size  = minimum_block_size
        self.minimum_block_size,
        self.sigma_signal = sigma_signal
        self.sigma_noise = sigma_noise
        self.random_state = random_state
    def _scale_matrix(self,
                      matrix: np.ndarray) -> np.ndarray:
        """
        Converts a covariance matrix into a correlation matrix.

        Args:
            covariance_matrix (np.ndarray): Input covariance matrix.

        Returns:
            np.ndarray: Output correlation matrix.
        """
        std = np.sqrt(np.diag(matrix))
        scaled_matrix = matrix / np.outer(std, std)
        scaled_matrix[scaled_matrix < -1] = -1  # numerical error
        scaled_matrix[scaled_matrix > 1] = 1  # numerical error
        return scaled_matrix

    def generate_sub_covariance(self,
                                number_observations: int,
                                number_factors: int, sigma: float,
                                random_state: Optional[int] = None) -> np.ndarray:
        """
        Generates a sub-covariance matrix from highly correlated factors.

        Args:
            number_observations (int): Number of observations of underlying factors.
            number_factors (int): Number of factors.
            sigma (float): Variance of added noise.
            random_state (Optional[int]): Random state.

        Returns:
            np.ndarray: Generated covariance matrix.
        """
        rng = check_random_state(random_state)
        if number_factors == 1:
            return np.ones((1, 1))
        data = rng.normal(size=(number_observations, 1))
        data = np.repeat(data, number_factors, axis=1)
        data += rng.normal(loc=0, scale=sigma, size=data.shape)
        covariance_matrix = np.cov(data, rowvar=False)
        return covariance_matrix

    def generate_random_block_covariance(self,
                                         number_factors: int,
                                         number_blocks: int,
                                         minimum_block_size: int = 1,
                                         sigma: float = 1.0,
                                         random_state: Optional[int] = None) -> np.ndarray:
        """
        Generates a random block covariance matrix.

        Args:
            number_factors (int): Number of factors building the covariance matrix.
            number_blocks (int): Number of blocks in the matrix.
            minimum_block_size (int, optional): Minimum size of each block. Defaults to 1.
            sigma (float, optional): Variance of the noise in the covariance blocks. Defaults to 1.0.
            random_state (Optional[int]): Random state.

        Returns:
            np.ndarray: The generated covariance matrix.
        """
        rng = check_random_state(random_state)
        parts = rng.choice(range(1, number_factors - (minimum_block_size - 1) * number_blocks), number_blocks - 1,
                           replace=False)
        parts.sort()
        parts = np.append(parts, number_factors - (minimum_block_size - 1) * number_blocks)
        parts = np.append(parts[0], np.diff(parts)) - 1 + minimum_block_size
        covariance_matrix = None
        for number_factors_ in parts:
            number_observations_ = int(max(number_factors_ * (number_factors_ + 1) / 2, 100))
            covariance_ = self.generate_sub_covariance(number_observations_,
                                                       number_factors_,
                                                       sigma,
                                                       random_state=rng)
            if covariance_matrix is None:
                covariance_matrix = covariance_.copy()
            else:
                covariance_matrix = block_diag(covariance_matrix, covariance_)
        return covariance_matrix

    def generate(self) -> pd.DataFrame:
        """
        Generates a random block correlation matrix.

        Args:
            number_factors (int): Number of factors building the covariance matrix.
            number_blocks (int): Number of blocks in the matrix.
            minimum_block_size (int, optional): Minimum size of each block. Defaults to 1.
            random_state (Optional[int]): Random state.

        Returns:
            pd.DataFrame: The generated correlation matrix.
        """
        rng = check_random_state(self.random_state)
        covariance_signal = self.generate_random_block_covariance(number_factors=self.number_factors,
                                                                  number_blocks= self.number_blocks,
                                                                  minimum_block_size=self.minimum_block_size,
                                                                  sigma=self.sigma_signal,
                                                                  random_state=rng)
        covariance_noise = self.generate_random_block_covariance(number_factors=self.number_factors,
                                                                 number_blocks=1,
                                                                 minimum_block_size=self.minimum_block_size,
                                                                 sigma=self.sigma_noise,
                                                                 random_state=rng)
        covariance_signal += covariance_noise
        correlation_matrix = self._scale_matrix(covariance_signal)
        return pd.DataFrame(correlation_matrix)

