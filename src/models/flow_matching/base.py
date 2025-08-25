# import tqdm
# import math
# import torch
# import numpy as np
# from torch import nn
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

# from typing import Optional, Literal, Tuple
# class FlowMatchingModel(nn.Module):
#     def __init__(self, 
#                  channels_data=32, 
#                  layers=5, 
#                  channels=512, 
#                  channels_t=512,
#                  lr=1e-4):
#         super().__init__()
#         self.channels_t = channels_t
#         self.channels_data = channels_data
        
#         # Network architecture
#         self.in_projection = nn.Linear(channels_data, channels)
#         self.t_projection = nn.Linear(channels_t, channels)
        
#         # Create blocks (equivalent to the old Block class)
#         blocks = []
#         for _ in range(layers):
#             blocks.append(nn.Linear(channels, channels))
#             blocks.append(nn.ReLU())
#         self.blocks = nn.Sequential(*blocks)
        
#         self.out_projection = nn.Linear(channels, channels_data)
        
#         # Initialize optimizer
#         # self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
#         self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
#         self.losses = []
    
#     def gen_t_embedding(self, t, max_positions=10000):
#         t = t * max_positions
#         half_dim = self.channels_t // 2
#         emb = math.log(max_positions) / (half_dim - 1)
#         emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
#         emb = t[:, None] * emb[None, :]
#         emb = torch.cat([emb.sin(), emb.cos()], dim=1)
#         if self.channels_t % 2 == 1:  # zero pad
#             emb = nn.functional.pad(emb, (0, 1), mode='constant')
#         return emb
    
#     def forward(self, x, t):
#         x = self.in_projection(x)
#         t = self.gen_t_embedding(t)
#         t = self.t_projection(t)
#         x = x + t 
#         x = self.blocks(x)
#         x = self.out_projection(x)
#         return x
    
#     def fit(self, data, training_steps=1000, batch_size=64):
#         """Training method for flow matching"""
#         self.train()
#         # pbar = tqdm.tqdm(range(training_steps))
        
#         for i in range(training_steps):
#             x1 = data[torch.randint(data.size(0), (batch_size,))]
#             x0 = torch.randn_like(x1)
#             target = x1 - x0
#             t = torch.rand(x1.size(0))
#             xt = (1 - t[:, None]) * x0 + t[:, None] * x1
#             pred = self(xt, t)
#             loss = ((target - pred)**2).mean()
#             loss.backward()
#             self.optimizer.step()
#             self.optimizer.zero_grad()
#             # pbar.set_postfix(loss=loss.item())
#             self.losses.append(loss.item())
#             if (i + 1) % 100 == 0:
#                 print(f"Epoch {i+1}/{training_steps}, Loss: {loss:.4f}")
    
#     def sample(self, num_samples=1500, steps=1000, seed=42):
#         """Sampling method for generation"""
#         torch.manual_seed(seed)
#         self.eval().requires_grad_(False)
        
#         xt = torch.randn(num_samples, self.channels_data)
        
#         for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
#             pred = self(xt, t.expand(xt.size(0)))
#             xt = xt + (1 / steps) * pred
        
#         self.train().requires_grad_(True)
#         print("Done Sampling")
#         return xt
        
import tqdm
import math
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from typing import Optional, Literal, Tuple
class FlowMatchingModel(nn.Module):
    def __init__(self, 
                 input_dimension: int=32, 
                 number_of_layers: int=5, 
                 hidden_dimension:int=512, 
                 time_dimension:int=512,
                 max_positions: int=10000,
                 learning_rate: float=1e-4):
        super().__init__()
        self.time_dimension = time_dimension
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.number_of_layers = number_of_layers
        self.learning_rate = learning_rate
        self.max_positions = max_positions
        
        # Network architecture
        self.input_projection = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.time_projection = nn.Linear(self.time_dimension, self.hidden_dimension)
        
        # Create blocks  
        blocks = []
        for _ in range(self.number_of_layers):
            blocks.append(nn.Linear(self.hidden_dimension, self.hidden_dimension))
            blocks.append(nn.ReLU())
        self.blocks = nn.Sequential(*blocks)
        
        self.output_projection = nn.Linear(self.hidden_dimension, self.input_dimension)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        self.losses = []
    
    def generate_time_embedding(self, 
                        t: torch.Tensor, 
                        max_positions:int=10000):
        t = t * max_positions
        half_dimension = self.time_dimension // 2
        embedding = math.log(max_positions) / (half_dimension - 1)
        embedding = torch.arange(half_dimension, device=t.device).float().mul(-embedding).exp()
        embedding = t[:, None] * embedding[None, :]
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=1)
        if self.time_dimension % 2 == 1:  # zero pad
            embedding = nn.functional.pad(embedding, (0, 1), mode='constant')
        return embedding
    
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor
                ):
        x = self.input_projection(x)
        t = self.generate_time_embedding(t, self.max_positions)
        t = self.time_projection(t)
        x = x + t 
        x = self.blocks(x)
        x = self.output_projection(x)
        return x
    
    def fit(self, 
            data: torch.Tensor,
            training_steps:int=1000,
            batch_size:int=64):
        """Training method for flow matching"""
        self.train()
        for i in range(training_steps):
            x1 = data[torch.randint(data.size(0), (batch_size,))]
            x0 = torch.randn_like(x1)
            target = x1 - x0
            t = torch.rand(x1.size(0))
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            pred = self(xt, t)
            loss = ((target - pred)**2).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.losses.append(loss.item())
            if (i + 1) % 100 == 0:
                print(f"Epoch {i+1}/{training_steps}, Loss: {loss:.4f}")
    
    def sample(self, 
               number_samples:int=1500, 
               number_steps:int=1000, 
               seed:int=42):
        """Sampling method for generation"""
        torch.manual_seed(seed)
        self.eval().requires_grad_(False)
        
        xt = torch.randn(number_samples, self.input_dimension)
        
        for i, t in enumerate(torch.linspace(0, 1, number_steps), start=1):
            pred = self(xt, t.expand(xt.size(0)))
            xt = xt + (1 / number_steps) * pred
        
        self.train().requires_grad_(True)
        print("Done Sampling")
        return xt
        
