from typing import Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import KFold

from src.models.rmt.KernelDensityEstimator import KernelDensityEstimator

class MarcenkoPastur:
    """
    Class for fitting the Marcenko-Pastur distribution to eigenvalues using Kernel Density Estimation (KDE).
    """
    def __init__(self,
                 grid_size: int = 1000,
                 kernel: str = 'gaussian',
                 cv: Union[int,BaseCrossValidator] = 100,
                 min_bandwidth_grid_exponent: int = -3,
                 max_bandwidth_grid_exponent: int = 1,
                 bandwidth_grid_size: int = 250,
                 initial_variance: float = 0.5,
                 epsilon: float = 1e-5,
                 min_q: float = 0.5,
                 max_q: float = 20,
                 q_grid_size: int = 20,
                 verbose: bool = True


    ):
        """
        Initialize the MarcenkoPastur object.

        Parameters:

        - grid_size: int, default=1000
            Number of data points in the grid for the Marcenko-Pastur probability density function.
        - kernel: str, default='gaussian'
            Kernel type for Kernel Density Estimation.
        - initial_variance: float, default=0.5
            Initial variance parameter for optimization.
        - epsilon: float, default=1e-5
            Small value used for bounds in optimization.
        """
        self.grid_size = grid_size
        self.kernel = kernel
        self.cv = cv
        self.min_bandwidth_grid_exponent = min_bandwidth_grid_exponent
        self.max_bandwidth_grid_exponent = max_bandwidth_grid_exponent
        self.bandwidth_grid_size = bandwidth_grid_size

        self.initial_variance = initial_variance
        self.epsilon = epsilon
        self.min_q  = min_q
        self.max_q  = max_q
        self.q_grid_size  = q_grid_size
        self.verbose = verbose
        self.q = None
        self.max_random_eigenvalue = None
        self.variance = None
        self.implied_variance = None
        self.estimated_number_signal_factors = None

    def get_pdf(self, q:float, variance: float) -> pd.Series:
        """
        Compute the probability density function of the Marcenko-Pastur distribution.

        Parameters:
        - q: float
            Ratio T/N where T is the number of rows and N is the number of columns of the data matrix.
        - variance: float
            Variance parameter of the Marcenko-Pastur distribution.

        Returns:
        - pdf: pd.Series
            Probability density function of the Marcenko-Pastur distribution.
        """
        min_eigenvalue = variance * (1 - np.sqrt(1. / q)) ** 2
        max_eigenvalue = variance * (1 + np.sqrt(1. / q)) ** 2
        eigenvalues_grid = np.linspace(min_eigenvalue, max_eigenvalue, self.grid_size)
        pdf = q / (2 * np.pi * variance * eigenvalues_grid) * (
                (max_eigenvalue - eigenvalues_grid) * (eigenvalues_grid - min_eigenvalue)) ** .5
        pdf = pd.Series(pdf, index=eigenvalues_grid)
        return pdf

    def empirical_vs_theoretical_pdf_error(self,
                                            variance: np.ndarray,

                                           kernel_density_estimator: np.ndarray) -> float:
        """
        Compute the sum of squared errors between theoretical and empirical distributions.

        Parameters:
        - variance: np.ndarray
            Variance parameter for the Marcenko-Pastur distribution.
        - kernel_density_estimator: KernelDensityEstimator
            Kernel Density Estimator object.

        Returns:
        - sse: float
            Sum of squared errors between theoretical and empirical distributions.
        """
        # print(self.q)
        theoretical_pdf = self.get_pdf(q=self.q, variance=variance[0] )
        # print(theoretical_pdf)
        empirical_pdf = kernel_density_estimator.evaluate_pdf(theoretical_pdf.index.values)
        sse = np.sum((theoretical_pdf - empirical_pdf) ** 2)
        return sse

    def fit(self, eigenvalues: np.ndarray) -> None:
        """
        Fit the Marcenko-Pastur distribution to eigenvalues using Kernel Density Estimation.

        Parameters:
        - eigenvalues: np.ndarray
            Eigenvalues of the data matrix.

        """
        kernel_density_estimator = KernelDensityEstimator(kernel=self.kernel,
                                                          cv=self.cv,
                                                          min_bandwidth_grid_exponent=self.min_bandwidth_grid_exponent,
                                                          max_bandwidth_grid_exponent=self.max_bandwidth_grid_exponent,
                                                          bandwidth_grid_size=self.bandwidth_grid_size)
        kernel_density_estimator.fit(observations=eigenvalues)
        q_grid = np.linspace(self.min_q, self.max_q, self.q_grid_size)
        out = {}
        min_fun = float(np.Infinity)
        best_q = None
        for q in q_grid:
            self.q =q
            result= minimize(lambda *x: self.empirical_vs_theoretical_pdf_error(*x, kernel_density_estimator),
                       x0=np.array([self.initial_variance]),
                       bounds=((self.epsilon, 1 - self.epsilon), )
                       )
            out[q] = result
            if result.fun <min_fun:
                min_fun = result['fun']
                best_q = q
            if self.verbose:
                max_random_eigenvalue = result['x'][0] * (1 + np.sqrt(1. / q)) ** 2
                estimated_number_signal_factors = eigenvalues.shape[0] - eigenvalues[::-1].searchsorted(
                    max_random_eigenvalue)
                print(f"For q={round(q,2)}, "
                      f"the optimal variance is {round(result['x'][0],2)} ,"
                      f" and objective function is {round(result['fun'],2)},"
                      f"and max random eigen value: {round(max_random_eigenvalue,2)},"
                      f"and # of signal factors: {round(estimated_number_signal_factors,2)}")
        if out[best_q]['success']:
            self.variance = out[best_q]['x'][0]
            self.q = best_q
        else:
            self.variance = 1
            self.q = 10

        self.max_random_eigenvalue = self.variance * (1 + np.sqrt(1. / self.q)) ** 2
        self.implied_variance = self.variance * (1 - self.max_random_eigenvalue / len(eigenvalues))


        self.estimated_number_signal_factors = np.sum(eigenvalues > self.max_random_eigenvalue)