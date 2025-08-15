

from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, Tuple, Union

from typing import List, Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from research.metrics import bin_summary_of_xy

def plot_bin_summary_of_xy(

        x: pd.Series,
        y: pd.Series,
        k:int,
        unique_flag : Optional[bool] = True,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,)->None:
    """ Calls the bin_summary_of_xy function and plots the returned bins."""
    bin_analytics = bin_summary_of_xy(x,y,k, unique_flag)
    plt.errorbar(

        x= bin_analytics["x_mean"],
        y=bin_analytics["y_mean"],
        yerr= bin_analytics["y_se"],
    )

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    return bin_analytics
def plot_heatmap(df: pd.DataFrame,
                 rows: List[str],
                 columns: List[str],
                 title: str,
                 fig_size: tuple)-> None:
    """Plot a heatmap on the specified columns and rows """
    plt.rcParams["figure.figsize"] = fig_size
    display_df = df[columns]
    display_df.index = rows
    sns.heatmap(display_df)
    plt.title(title)
    plt.show()
    plt.close()


def qqplot(x: Union[np.ndarray, list],
           y: Union[np.ndarray, list],
           quantiles: Optional[np.ndarray] = None,
           interpolation: str = 'linear',
           ax: Optional[plt.Axes] = None,
           figsize: Tuple[int, int] = (8, 6),
           title: str = "Q-Q Plot",
           xlabel: str = "X Quantiles",
           ylabel: str = "Y Quantiles",
           show_stats: bool = True,
           reference_line: bool = True,
           alpha: float = 0.6,
           s: float = 20) -> plt.Axes:
    """
    Create a quantile-quantile (Q-Q) plot comparing two distributions.

    Parameters:
    -----------
    x : array-like
        First dataset
    y : array-like
        Second dataset
    quantiles : array-like, optional
        Quantiles to compute. If None, uses evenly spaced quantiles from 0.01 to 0.99
    interpolation : str, default 'linear'
        Interpolation method for quantile calculation
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default (8, 6)
        Figure size if creating new figure
    title : str, default "Q-Q Plot"
        Plot title
    xlabel : str, default "X Quantiles"
        X-axis label
    ylabel : str, default "Y Quantiles"
        Y-axis label
    show_stats : bool, default True
        Whether to show correlation and other statistics
    reference_line : bool, default True
        Whether to show y=x reference line
    alpha : float, default 0.6
        Point transparency
    s : float, default 20
        Point size

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object containing the plot

    Example:
    --------
    >>> import numpy as np
    >>> x = np.random.normal(0, 1, 1000)
    >>> y = np.random.normal(0.5, 1.2, 1000)
    >>> ax = qqplot(x, y, title="Normal vs Normal (shifted)")
    >>> plt.show()
    """

    # Convert to numpy arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # Remove NaN values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Define quantiles if not provided
    if quantiles is None:
        n_quantiles = min(len(x), len(y), 100)  # Use up to 100 quantiles
        quantiles = np.linspace(0.01, 0.99, n_quantiles)

    # Calculate quantiles for both distributions
    x_quantiles = np.quantile(x, quantiles, method=interpolation)
    y_quantiles = np.quantile(y, quantiles, method=interpolation)

    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create the Q-Q plot
    ax.scatter(x_quantiles, y_quantiles, alpha=alpha, s=s,
               color='steelblue', edgecolors='navy', linewidth=0.5)

    # Add reference line (y=x)
    if reference_line:
        min_val = min(np.min(x_quantiles), np.min(y_quantiles))
        max_val = max(np.max(x_quantiles), np.max(y_quantiles))
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', alpha=0.8, linewidth=2, label='y = x')
        ax.legend()

    # Calculate and display statistics
    if show_stats:
        correlation = np.corrcoef(x_quantiles, y_quantiles)[0, 1]

        # Calculate Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(x, y)

        # Add statistics text box
        stats_text = f'Correlation: {correlation:.3f}\n'
        stats_text += f'KS statistic: {ks_stat:.3f}\n'
        stats_text += f'KS p-value: {ks_pvalue:.3f}'

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                                                   facecolor='wheat', alpha=0.8))

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Make plot square and add grid
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    return ax


def qqplot_against_normal(data: Union[np.ndarray, list],
                          ax: Optional[plt.Axes] = None,
                          figsize: Tuple[int, int] = (8, 6),
                          title: str = "Q-Q Plot vs Normal Distribution",
                          **kwargs) -> plt.Axes:
    """
    Create a Q-Q plot comparing data against a normal distribution.

    Parameters:
    -----------
    data : array-like
        Dataset to compare against normal distribution
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
    figsize : tuple, default (8, 6)
        Figure size if creating new figure
    title : str, default "Q-Q Plot vs Normal Distribution"
        Plot title
    **kwargs : dict
        Additional arguments passed to qqplot function

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    """

    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]

    # Generate normal distribution with same mean and std
    normal_data = np.random.normal(np.mean(data), np.std(data), len(data))

    return qqplot(normal_data, data, ax=ax, figsize=figsize, title=title,
                  xlabel="Theoretical Normal Quantiles",
                  ylabel="Sample Quantiles", **kwargs)





def plot_results(
    original_data: np.ndarray, 
    generated_data: np.ndarray, 
    title: str = "Results"
) -> None:
    """Plot original vs generated data
    
    Args:
        original_data: Original dataset as numpy array
        generated_data: Generated dataset as numpy array  
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original data
    ax1.scatter(original_data[:, 0], original_data[:, 1], alpha=0.6, s=20)
    ax1.set_title("Original Spiral Data")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Generated data
    ax2.scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.6, s=20, color='red')
    ax2.set_title("Generated Data")
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



# Example usage and demonstration
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)

    # Example 1: Two normal distributions
    x1 = np.random.normal(0, 1, 1000)
    y1 = np.random.normal(0.5, 1.2, 1000)

    # Example 2: Normal vs exponential
    x2 = np.random.normal(0, 1, 1000)
    y2 = np.random.exponential(1, 1000)

    # Example 3: Two similar distributions
    x3 = np.random.normal(0, 1, 1000)
    y3 = np.random.normal(0.1, 1.1, 1000)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot examples
    qqplot(x1, y1, ax=axes[0, 0], title="Normal vs Normal (shifted)")
    qqplot(x2, y2, ax=axes[0, 1], title="Normal vs Exponential")
    qqplot(x3, y3, ax=axes[1, 0], title="Similar Normal Distributions")
    qqplot_against_normal(y2, ax=axes[1, 1], title="Exponential vs Normal")

    plt.tight_layout()
    plt.show()

    # Print interpretation guide
    print("\nQQ Plot Interpretation Guide:")
    print("=" * 50)
    print("• Points close to y=x line → distributions are similar")
    print("• S-shaped curve → one distribution has heavier tails")
    print("• Inverted S-shape → one distribution has lighter tails")
    print("• Linear but not y=x → different location/scale parameters")
    print("• High correlation → strong linear relationship between quantiles")
    print("• KS test p-value < 0.05 → distributions significantly different")

