from typing import Union
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf


def correlation2covariance(correlation_matrix: np.ndarray,
                           standard_deviations: np.ndarray) -> np.ndarray:
    covariance_matrix = correlation_matrix * np.outer(standard_deviations, standard_deviations)
    return covariance_matrix


def build_block_matrix(number_blocks: int,
                       block_size: int,
                       block_correlation: float) -> np.ndarray:
    block = np.ones((block_size, block_size)) * block_correlation
    block[range(block_size), range(block_size)] = 1
    matrix = block_diag(*([block] * number_blocks))
    return matrix


def build_true_matrix(number_blocks: int,
                      block_size: int,
                      block_correlation: float,
                      ) -> Union[np.ndarray, np.ndarray]:
    block_matrix = build_block_matrix(number_blocks, block_size, block_correlation)
    block_matrix = pd.DataFrame(block_matrix)
    column_names = block_matrix.columns.tolist()
    np.random.shuffle(column_names)
    block_matrix = block_matrix[column_names].loc[column_names].copy(deep=True)
    standard_deviations = np.random.uniform(.05, .2, block_matrix.shape[0])
    covariance_matrix = correlation2covariance(block_matrix, standard_deviations)
    mu = np.random.normal(standard_deviations, standard_deviations**(0.5), covariance_matrix.shape[0]).reshape(-1, 1)
    return mu, covariance_matrix


def simulate_covariance_mean(true_mu: np.ndarray,
                             true_covariance: np.ndarray,
                             number_of_observations: int,
                             shrink: bool = False) -> Union[np.ndarray, np.ndarray]:
    sample_data = np.random.multivariate_normal(true_mu.flatten(),
                                         true_covariance,
                                         size=number_of_observations)

    sample_mu = sample_data.mean(axis=0).reshape(-1, 1)


    if shrink:
        sample_covariance = LedoitWolf().fit(sample_data).covariance_
    else:
        sample_covariance = np.cov(sample_data, rowvar=0)
    return sample_data, sample_mu, sample_covariance
