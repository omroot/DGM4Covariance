
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import learning_curve ,GridSearchCV
from sklearn.model_selection import LeaveOneOut, KFold


def compute_pca(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    eigenvalues,eigenvectors=np.linalg.eigh(matrix)

    indices=eigenvalues.argsort()[::-1] # arguments for sorting eVal desc
    eigenvalues =eigenvalues[indices]
    eigenvalues=np.diagflat(eigenvalues)

    eigenvectors = eigenvectors[:,indices]
    return eigenvalues,eigenvectors

def covariance2correlation(covariance_matrix: np.ndarray)-> np.ndarray:
    """Converts a covariance matrix into a correlation matrix
    Args:
        covariance_matrix: input covariance matrix
    Returns:
        correlation_matrix: output correlation matrix
    """
    # Derive the correlation matrix from a covariance matrix
    std=np.sqrt(np.diag(covariance_matrix))
    correlation_matrix=covariance_matrix/np.outer(std,std)
    correlation_matrix[correlation_matrix<-1] = -1 # numerical error
    correlation_matrix[correlation_matrix>1]  = 1 # numerical error
    return correlation_matrix

def correlation2covariance(correlation_matrix: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Converts a correlation matrix into a covariance matrix.
    
    Args:
        correlation_matrix: input correlation matrix (NxN)
        std: array of standard deviations of the variables (length N)
        
    Returns:
        covariance_matrix: output covariance matrix (NxN)
    """
    covariance_matrix = correlation_matrix * np.outer(std, std)
    return covariance_matrix