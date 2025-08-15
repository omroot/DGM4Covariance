from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity

class KernelDensityEstimator:
    def __init__(self,
                 kernel: str = 'gaussian',
                 kernel_bandwidth: float = None,
                 cv: Union[int, BaseCrossValidator] = 100 ,
                 min_bandwidth_grid_exponent: int = -4,
                 max_bandwidth_grid_exponent: int = -1,
                 bandwidth_grid_size: int = 100):

        """
        Initialize the KernelDensityEstimator class.

        Parameters:
        - kernel: str, default='gaussian'
            The kernel function to be used. Valid values are 'gaussian', 'tophat', 'epanechnikov', 'exponential',
            'linear', 'cosine'.
        - kernel_bandwidth: float, default=None
            The bandwidth of the kernel. If None, it will be determined automatically during fitting.
        - cv: BaseCrossValidator, default=KFold(100)
            The cross-validation strategy used during bandwidth optimization.
        - min_bandwidth_grid_exponent: int, default=-3
            The minimum exponent of the bandwidth grid for optimization.
        - max_bandwidth_grid_exponent: int, default=1
            The maximum exponent of the bandwidth grid for optimization.
        - bandwidth_grid_size: int, default=250
            The number of bandwidth values to consider during optimization.
        """
        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.cv = cv
        self.KernelDensityEstimator = None
        self.min_bandwidth_grid_exponent = min_bandwidth_grid_exponent
        self.max_bandwidth_grid_exponent = max_bandwidth_grid_exponent
        self.bandwidth_grid_size = bandwidth_grid_size

    def find_optimal_kernel_bandwidth(self,
                                      observations: np.ndarray) -> None:
        """
        Find the optimal kernel bandwidth value using grid search.

        Parameters:
        - observations: np.ndarray
            Input observations used for bandwidth optimization.
        """
        bandwidths = 10 ** np.linspace(self.min_bandwidth_grid_exponent,
                                       self.max_bandwidth_grid_exponent,
                                       self.bandwidth_grid_size)
        grid = GridSearchCV(estimator = KernelDensity(kernel=self.kernel),
                            param_grid= {'bandwidth': bandwidths},
                            cv=self.cv)
        grid.fit(observations)
        self.kernel_bandwidth = grid.best_params_['bandwidth']

    def fit(self, observations: np.ndarray) -> None:
        """
        Fit the KernelDensity estimator.

        Parameters:
        - observations: np.ndarray
            Input observations to fit the estimator.
        """
        if len(observations.shape) == 1:
            observations = observations.reshape(-1, 1)
        if self.kernel_bandwidth is None:
            self.find_optimal_kernel_bandwidth(observations=observations)
        self.KernelDensityEstimator = KernelDensity(kernel=self.kernel, bandwidth=self.kernel_bandwidth).fit(observations)

    def evaluate_pdf(self, evaluation_observations=None) -> pd.Series:
        """
        Evaluate the probability density function (pdf) on evaluation observations.

        Parameters:
        - evaluation_observations: np.ndarray or None, default=None
            Optional array of observations to evaluate the pdf. If None, an error will be raised.

        Returns:
        - pdf: pd.Series
            Probability density function evaluated on the evaluation observations.
        """
        if evaluation_observations is None:
            raise ValueError("Evaluation observations must be provided.")
        if len(evaluation_observations.shape) == 1:
            evaluation_observations = evaluation_observations.reshape(-1, 1)
        logProb = self.KernelDensityEstimator.score_samples(evaluation_observations)  # log(density)
        pdf = pd.Series(np.exp(logProb), index=evaluation_observations.flatten())
        return pdf
