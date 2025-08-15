import numpy as np

class MatrixGenerator:
    """
    A class for generating covariance matrices with signal and noise components.
    """

    def __init__(self,
                 number_total_factors: int = 1000,
                 number_signal_factors: int = 100
                 ):
        """
        Initialize the MatrixGenerator.

        Parameters:
        - noise_proportion: float, default=0.95
            Proportion of noise in the generated covariance matrix.
        - number_total_factors: int, default=1000
            Total number of factors in the covariance matrix.
        - number_signal_factors: int, default=100
            Number of signal factors contributing to the covariance matrix.
        """
        self.number_total_factors = number_total_factors
        self.number_signal_factors = number_signal_factors

    def generate_noise_covariance(self,
                                  q: int = 10) -> np.ndarray:
        """
        Generates a random noise covariance matrix.

        Parameters:
        - q: int, default=10
            Scaling factor for noise covariance matrix.

        Returns:
        - noise_covariance_matrix: np.ndarray
            The generated noise covariance matrix.
        """
        noise_covariance_matrix = np.cov(np.random.normal(size=(self.number_total_factors * q,
                                                                self.number_total_factors)),
                                         rowvar=0)
        return noise_covariance_matrix

    def generate_signal_covariance(self) -> np.ndarray:
        """
        Generates a random signal covariance matrix.

        Returns:
        - signal_covariance_matrix: np.ndarray
            The generated signal covariance matrix.
        """
        w = np.random.normal(size=(self.number_total_factors, self.number_signal_factors))
        # Random cov matrix, not full rank
        signal_covariance_matrix = np.dot(w, w.T)
        # Full rank cov (linearly independent rows and columns)
        signal_covariance_matrix += np.diag(np.random.uniform(size=self.number_total_factors))
        return signal_covariance_matrix

    def generate(self,
                 noise_proportion: float = 0.95,
                 q: int = 10) -> np.ndarray:
        """
        Generates a combined covariance matrix with signal and noise components.

        Parameters:
        - noise_proportion: float, default=0.95
            Proportion of noise in the generated covariance matrix.
        - q: int, default=10
            Scaling factor for noise covariance matrix.

        Returns:
        - combined_covariance_matrix: np.ndarray
            The generated combined covariance matrix.
        """

        noise_covariance_matrix = self.generate_noise_covariance(q=q)
        signal_covariance_matrix = self.generate_signal_covariance()
        combined_covariance_matrix = noise_proportion * noise_covariance_matrix + \
                                     (1 - noise_proportion) * signal_covariance_matrix
        return combined_covariance_matrix
