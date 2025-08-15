import numpy as np


def get_maximum_sharpe_portfolio_weights(covariance_matrix: np.ndarray,
                                        mu: np.ndarray)-> np.ndarray:
    """Computes the maximum sharpe portfolio weights """
    inverse_matrix = np.linalg.inv(covariance_matrix)
    portfolio_weights=np.dot(inverse_matrix, mu)
    ones = np.ones(shape = (inverse_matrix.shape[0],1))
    portfolio_weights = portfolio_weights/np.dot(ones.T, portfolio_weights)
    return portfolio_weights
