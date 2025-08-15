
from itertools import groupby
from operator import itemgetter
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.utils import check_random_state
from scipy.linalg import block_diag

def shuffle_matrix(input_matrix: pd.DataFrame)->pd.DataFrame:
    new_indicies = list(range(input_matrix.shape[0]))
    random.shuffle(new_indicies)
    shuffled_matrix = input_matrix.iloc[new_indicies]  # reorder rows
    shuffled_matrix = shuffled_matrix.iloc[:, new_indicies]  # reorder columns
    return shuffled_matrix

def generate_sub_covariance(number_observations: int,
                              number_factors: int,
                              sigma: float,
                              random_state=None) -> np.ndarray:
    """"Generate a sub correlation matrix from highly correlated factors
    Args:
        number_observations: number of observations of underlying factors
        number_factors: number of factors
        sigma: variance of added noise
        random_state: random state

    Returns:
        covariance_matrix: generated covariance matrix
    """
    rng = check_random_state(random_state)
    if number_factors == 1:
        return np.ones((1,1))
    data = rng.normal(size=(number_observations, 1)) #array of normal rv
    data = np.repeat(data, number_factors, axis=1) #matrix of columns repeating rv. Simulate time-series of at least 100 elements.
    data += rng.normal(loc=0, scale=sigma, size=data.shape) #add two rv X~Y~N(0,1), Z=X+Y~N(0+0, 1+1)=N(0,2)
    covariance_matrix = np.cov(data, rowvar=False) #ar0.shape = nCols x nCols
    return covariance_matrix


def generate_random_block_covariance(number_factors: int,
                                     number_blocks: int,
                                     minimum_block_size:int =1,
                                     sigma:float =1.,
                                     random_state=None) -> np.ndarray:
    """ Generate a random block covariance matrix
    Args:
        number_factors: number of factors building the covariance matrix
        number_blocks: number of blocks in the matrix
        minimum_block_size: minimum size of each block
        sigma: variance of the noise in the covariance blocks
        random_state: random state
    Returns:
        covariance_matrix: the generated covariance matrix

    """
    rng = check_random_state(random_state)
    parts = rng.choice(range(1, number_factors - (minimum_block_size - 1) * number_blocks), number_blocks - 1, replace=False)
    parts.sort()
    parts = np.append(parts, number_factors - (minimum_block_size - 1) * number_blocks)  # add nCols to list of parts, unless minBlockSize>1
    parts = np.append(parts[0], np.diff(parts)) - 1 + minimum_block_size
    print("block sizes:" + str(parts))
    covariance_matrix = None
    for number_factors_ in parts:
        # number of observations the underlying factors are generated from
        number_observationsـ = int(max(number_factors_ * (number_factors_ + 1) / 2., 100))
        covariance_ = generate_sub_covariance(number_observationsـ,
                                               number_factors_,
                                               sigma,
                                               random_state=rng)
        if covariance_matrix is None:
            covariance_matrix = covariance_.copy()
        else:
            covariance_matrix = block_diag(covariance_matrix, covariance_)  # list of square matrix on larger matrix on the diagonal

    return covariance_matrix


def generate_random_block_correlation_matrix(number_factors: int,
                                             number_blocks: int ,
                                             minimum_block_size: int=1,
                                             random_state=None
                                             ):
    """Generate a random block correlation matrix
    Args:
        number_factors: number of factors building the covariance matrix
        number_blocks: number of blocks in the matrix
        minimum_block_size: minimum size of each block
        sigma: variance of the noise in the covariance blocks
        random_state: random state
    Returns:
        covariance_matrix: the generated covariance matrix
    """
    rng = check_random_state(random_state)

    covariance_signal = generate_random_block_covariance(number_factors,
                                            number_blocks,
                                            minimum_block_size=minimum_block_size,
                                            sigma=.5,
                                            random_state=rng)
    # generate noise
    covariance_noise = generate_random_block_covariance(number_factors,
                                            1,
                                            minimum_block_size=minimum_block_size,
                                            sigma=1.,
                                            random_state=rng)
    # add noise
    covariance_signal += covariance_noise
    correlation_matrix = covariance2correlation(covariance_signal)
    correlation_df = pd.DataFrame(correlation_matrix)
    return correlation_df

