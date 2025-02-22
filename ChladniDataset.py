from typing import Tuple, Union
import torch
import os
import requests
import numpy as np
import scipy.io
from FunctionEncoder.Dataset.BaseDataset import BaseDataset

class ChladniDataset(BaseDataset):
    def __init__(self,
                 file_url: str = "https://drive.usercontent.google.com/download?id=1pI3AGj7RIAEX4INmMyF0cGu_91hlqWRf&export=download&authuser=0&confirm=t&uuid=75e1032b-0325-4bae-9adf-ca5a22218842&at=AEz70l6gQs9gqm0YryFo31bI0uOH:1740260353274",
                 local_file: str = "ChladniData.mat",
                 n_functions_per_sample: int = 10,
                 n_examples_per_sample: int = 100,
                 n_points_per_sample: int = 1000,
                 device: str = "auto",
                 dtype: torch.dtype = torch.float32):
        # Set input size to (2,) for (x,y) coordinate pairs and output size to (1,)
        super().__init__(input_size=(2,),
                         output_size=(1,),
                         total_n_functions=float('inf'),
                         total_n_samples_per_function=float('inf'),
                         data_type="deterministic",
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         device=device,
                         dtype=dtype)
        
        # Download the dataset file if not already present
        if not os.path.exists(local_file):
            print(f"Downloading dataset from {file_url} to {local_file}...")
            r = requests.get(file_url, stream=True)
            with open(local_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download complete.")
        
        # Load the .mat file; expected keys: 'alpha_train', 'S_train', 'x', 'y'
        mat = scipy.io.loadmat(local_file)
        alpha_train = mat['alpha_train']  # shape: (n_range, m_range, N_train)
        S_train = mat['S_train']          # shape: (numPoints, numPoints, N_total)
        x = mat['x'].squeeze()            # vector of grid x-coordinates (length = numPoints)
        y = mat['y'].squeeze()            # vector of grid y-coordinates (length = numPoints)
        
        # Reformat and convert data to torch tensors on the correct device and dtype.
        # Permute alpha_train so that training samples are along the first dimension:
        # From (n_range, m_range, N_train) to (N_train, n_range, m_range)
        self.alpha_train = torch.tensor(alpha_train, dtype=self.dtype, device=self.device).permute(2, 0, 1)
        
        # Permute S_train from (numPoints, numPoints, N_total) to (N_total, numPoints, numPoints)
        S_train = torch.tensor(S_train, dtype=self.dtype, device=self.device).permute(2, 0, 1)
        # Add a channel dimension so that each output is a scalar:
        self.S_train = S_train.unsqueeze(-1)  # shape: (N_total, numPoints, numPoints, 1)
        
        # Create a grid of (x,y) coordinate pairs from the provided x and y vectors.
        # First, build a meshgrid, then stack and flatten.
        X, Y = torch.meshgrid(torch.tensor(x, dtype=self.dtype, device=self.device),
                              torch.tensor(y, dtype=self.dtype, device=self.device), indexing='ij')
        grid = torch.stack((X, Y), dim=-1)  # shape: (numPoints, numPoints, 2)
        grid = grid.view(-1, 2)             # shape: (total_points, 2)
        self.grid = grid
        self.num_points = grid.shape[0]
        
        # Total number of training samples available
        self.N_train = self.alpha_train.shape[0]

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
          - example_xs: (n_functions, n_examples, 2) coordinate pairs for support points.
          - S_train (support values): (n_functions, n_examples, 1) corresponding forcing values.
          - xs: (n_functions, n_points, 2) coordinate pairs for query points.
          - S_train (query values): (n_functions, n_points, 1) corresponding forcing values.
          - dict: A dictionary with key "alpha_train" containing the alpha coefficients for each function.
        """
        with torch.no_grad():
            n_functions = self.n_functions_per_sample
            n_examples = self.n_examples_per_sample
            n_points = self.n_points_per_sample
            
            # Randomly select n_functions training samples
            indices = torch.randint(0, self.N_train, (n_functions,))
            
            total_points = self.num_points
            # Determine whether to sample with replacement if the requested number exceeds total points
            replace_examples = n_examples > total_points
            replace_points = n_points > total_points
            
            # For each function, randomly sample indices from the grid
            example_indices = torch.stack([
                torch.tensor(np.random.choice(total_points, n_examples, replace=replace_examples))
                for _ in range(n_functions)
            ], dim=0).to(self.device)
            query_indices = torch.stack([
                torch.tensor(np.random.choice(total_points, n_points, replace=replace_points))
                for _ in range(n_functions)
            ], dim=0).to(self.device)
            
            # Gather coordinate pairs for the support and query sets.
            example_xs = self.grid[example_indices]  # shape: (n_functions, n_examples, 2)
            query_xs = self.grid[query_indices]        # shape: (n_functions, n_points, 2)
            
            # Flatten the spatial dimensions of S_train so that we can index it with grid indices.
            # S_train has shape (N_train, numPoints, numPoints, 1) -> (N_train, total_points, 1)
            S_train_flat = self.S_train.view(self.N_train, -1, 1)
            
            # For each selected function sample, extract the corresponding S_train values.
            example_S = torch.stack([
                S_train_flat[idx][example_indices[i]] for i, idx in enumerate(indices)
            ], dim=0)
            query_S = torch.stack([
                S_train_flat[idx][query_indices[i]] for i, idx in enumerate(indices)
            ], dim=0)
            
            # Gather the corresponding alpha coefficients.
            alpha = self.alpha_train[indices]  # shape: (n_functions, n_range, m_range)
            
            return example_xs, example_S, query_xs, query_S, {"alpha_train": alpha}