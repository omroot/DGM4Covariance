import numpy as np
def get_minimum_variance_portfolio_weights(covariance_matrix: np.ndarray) -> np.ndarray:
    """Computes the minimum variance portfolio weights """
    inverse_matrix = np.linalg.inv(covariance_matrix)
    ones = np.ones(shape=(inverse_matrix.shape[0], 1))
    portfolio_weights = np.dot(inverse_matrix, ones)
    portfolio_weights = portfolio_weights / np.dot(ones.T, portfolio_weights)
    return portfolio_weights

