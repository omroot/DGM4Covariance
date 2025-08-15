"""
Simple Vanilla DDPM for Tabular Data - Spiral Example
Minimal implementation focusing on core DDPM concepts
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class ScoreNet(nn.Module):
    """Simple MLP for denoising"""
    def __init__(self, input_dim=2, hidden_dim=128, time_dim=32):
        super().__init__()
        
        # Time embedding (simplified)
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        # Simple time encoding
        t_normalized = t.float().unsqueeze(-1) / 1000.0  # Normalize timestep
        t_emb = self.time_embed(t_normalized)
        
        # Concatenate input and time embedding
        x_with_time = torch.cat([x, t_emb], dim=-1)
        return self.net(x_with_time)
    


class DDPM:
    def __init__(self, timesteps=4000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For sampling
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
    
    def add_noise(self, x_0, t):
        """Forward diffusion: add noise to data"""
        noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1)
        
        # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise
    
    def sample(self, model, n_samples, input_dim=32):
        """Reverse diffusion: generate samples"""
        model.eval()
        
        # Start from pure noise
        x = torch.randn(n_samples, input_dim).to(device)
        
        with torch.no_grad():
            for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
                # Current timestep for all samples
                t_tensor = torch.full((n_samples,), t, dtype=torch.long).to(device)
                
                # Predict noise
                predicted_noise = model(x, t_tensor)
                
                # Compute coefficients
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alpha_cumprod[t]
                beta_t = self.betas[t]
                
                # Compute mean of reverse distribution
                x_mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise
                )
                
                if t > 0:
                    # Add noise (except for last step)
                    noise = torch.randn_like(x)
                    x = x_mean + torch.sqrt(beta_t) * noise
                else:
                    x = x_mean
        
        model.train()
        return x
    

from typing import List, Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def train_ddpm(
    model: torch.nn.Module, 
    data: np.ndarray, 
    ddpm: Any,  # DDPM class with .timesteps attribute and .add_noise() method
    epochs: int = 10000, 
    batch_size: int = 128, 
    lr: float = 1e-3
) -> List[float]:
    """Train the DDPM model
    
    Args:
        model: PyTorch neural network model
        data: Training data as numpy array
        ddpm: DDPM diffusion model instance
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate for optimizer
        
    Returns:
        List of average losses per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    # Assuming device is defined globally, otherwise add device parameter
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert data to tensor
    data_tensor = torch.from_numpy(data).to(device)
    n_samples = len(data_tensor)
    
    losses: List[float] = []
    
    for epoch in range(epochs):
        epoch_losses: List[float] = []
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            batch = data_tensor[i:i+batch_size]
            
            # Sample random timesteps
            t = torch.randint(0, ddpm.timesteps, (len(batch),)).to(device)
            
            # Add noise (fixed typo: ddmp -> ddpm)
            x_t, noise = ddpm.add_noise(batch, t)
            
            # Predict noise
            predicted_noise = model(x_t, t)
            
            # Compute loss (simple MSE)
            loss = nn.MSELoss()(predicted_noise, noise)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss: float = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses