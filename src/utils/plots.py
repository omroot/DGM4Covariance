
from typing import List, Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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