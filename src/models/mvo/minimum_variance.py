import numpy as np
def get_minimum_variance_portfolio_weights(covariance_matrix: np.ndarray) -> np.ndarray:
    """Computes the minimum variance portfolio weights """
    inverse_matrix = np.linalg.inv(covariance_matrix)
    ones = np.ones(shape=(inverse_matrix.shape[0], 1))
    portfolio_weights = np.dot(inverse_matrix, ones)
    portfolio_weights = portfolio_weights / np.dot(ones.T, portfolio_weights)
    return portfolio_weights

import numpy as np

def get_inverse_variance_portfolio_weights(covariance_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the inverse-variance (IV) portfolio weights.

    Parameters
    ----------
    covariance_matrix : np.ndarray
        (N x N) covariance matrix of asset returns.

    Returns
    -------
    np.ndarray
        (N x 1) column vector of fully-invested IV weights:
        w_i ∝ 1 / Var_i, normalized to sum to 1.

    Notes
    -----
    Uses only the diagonal of the covariance matrix (ignores correlations).
    """
    # basic shape checks
    if covariance_matrix.ndim != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance_matrix must be square (N x N).")

    variances = np.diag(covariance_matrix).astype(float)
    if np.any(variances <= 0):
        raise ValueError("All variances must be positive for inverse-variance weights.")

    # D^{-1} 1 / (1^T D^{-1} 1)
    D_inv = np.diag(1.0 / variances)
    ones = np.ones((covariance_matrix.shape[0], 1))
    weights = D_inv @ ones
    weights = weights / (ones.T @ weights)

    return weights
