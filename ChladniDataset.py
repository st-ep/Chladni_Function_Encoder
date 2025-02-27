from typing import Tuple, Union
import torch
import os
import numpy as np
import scipy.io
import requests
from FunctionEncoder.Dataset.BaseDataset import BaseDataset
import h5py


class ChladniDataset(BaseDataset):
    def __init__(self,
                 n_functions: int = 100,      # Default that works with GPU memory
                 n_examples: int = 625,    # Default for good function approximation
                 file_url: str = "https://drive.usercontent.google.com/download?id=1pI3AGj7RIAEX4INmMyF0cGu_91hlqWRf&export=download&authuser=0&confirm=t&uuid=cfddba53-15e0-464f-833e-043384ca6cb9&at=AEz70l70Rq_vh6_2s32D4mztlm-k:1740298306827",
                 local_file: str = "ChladniData.mat",
                 device: str = "auto",
                 dtype: torch.dtype = torch.float32):
        # Initialize parent class with correct parameters
        super().__init__(input_size=(2,),
                         output_size=(1,),
                         data_type="deterministic",
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=625)  # Set n_queries to match grid size (25x25=625)
        
        # Rest of the initialization code...
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dtype = dtype
        
        # Download the dataset file if not already present
        if not os.path.exists(local_file):
            print(f"Downloading dataset from {file_url} to {local_file}...")
            r = requests.get(file_url, stream=True)
            with open(local_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download complete.")
        
        # Load the .mat file using h5py
        with h5py.File(local_file, 'r') as f:
            alpha_train = np.array(f['alpha_train']).transpose()
            S_train = np.array(f['S_train']).transpose()
            x = np.array(f['x']).squeeze()
            y = np.array(f['y']).squeeze()
        
        # Convert to tensors
        self.alpha_train = torch.tensor(alpha_train, dtype=self.dtype, device=self.device)
        S_train = torch.tensor(S_train, dtype=self.dtype, device=self.device)
        
        # Store original statistics for reference
        self.S_mean = S_train.mean()
        self.S_std = S_train.std()
        
        # Permute and reshape without normalization
        S_train = S_train.permute(2, 0, 1)
        S_train = S_train.contiguous()
        self.S_train = S_train.unsqueeze(-1)
        
        # Create grid
        X, Y = torch.meshgrid(torch.tensor(x, dtype=self.dtype, device=self.device),
                             torch.tensor(y, dtype=self.dtype, device=self.device),
                             indexing='ij')
        grid = torch.stack((X, Y), dim=-1)
        grid = grid.view(-1, 2)
        
        # Normalize grid coordinates to [-1, 1]
        self.grid_min = grid.min(dim=0)[0]
        self.grid_max = grid.max(dim=0)[0]
        self.grid = 2 * (grid - self.grid_min) / (self.grid_max - self.grid_min) - 1
        
        self.num_points = self.grid.shape[0]
        self.N_train = self.alpha_train.shape[0]
        
        # Store grid dimensions for reference
        self.grid_size = (len(x), len(y))
        
        # Add data validation checks after loading
        print(f"Dataset Statistics:")
        print(f"S_train shape: {self.S_train.shape}")
        print(f"Grid shape: {self.grid.shape}")
        print(f"Grid dimensions: {self.grid_size[0]}x{self.grid_size[1]} ({self.num_points} total points)")
        
        # Print value ranges
        print("\nValue ranges:")
        print(f"Alpha train: [{self.alpha_train.min():.3f}, {self.alpha_train.max():.3f}]")
        print(f"S_train: [{self.S_train.min():.3f}, {self.S_train.max():.3f}]")
        print(f"Grid x: [{self.grid[:, 0].min():.3f}, {self.grid[:, 0].max():.3f}]")
        print(f"Grid y: [{self.grid[:, 1].min():.3f}, {self.grid[:, 1].max():.3f}]")

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
          - example_xs: (n_functions, n_examples, 2) coordinate pairs for support points.
          - example_S: (n_functions, n_examples, 1) corresponding forcing values.
          - query_xs: (n_functions, n_queries, 2) coordinate pairs for query points.
          - query_S: (n_functions, n_queries, 1) corresponding forcing values.
          - dict: A dictionary with key "alpha_train" containing the alpha coefficients for each function.
        """
        with torch.no_grad():
            n_functions = self.n_functions
            n_examples = self.n_examples
            n_queries = self.n_queries
            
            # Randomly select n_functions training samples
            indices = torch.randint(0, self.N_train, (n_functions,))
            
            total_points = self.num_points  # This should be 625 (25x25)
            # Determine whether to sample with replacement if requested number exceeds total points
            replace_examples = n_examples > total_points
            
            # For each function, randomly sample indices for examples
            example_indices = torch.stack([
                torch.tensor(np.random.choice(total_points, n_examples, replace=replace_examples))
                for _ in range(n_functions)
            ], dim=0).to(self.device)
            
            # For queries, either use all grid points or sample a subset
            if n_queries == total_points:
                # Use all grid points
                query_indices = torch.arange(total_points).unsqueeze(0).expand(n_functions, -1).to(self.device)
            else:
                # Sample a subset of grid points
                replace_queries = n_queries > total_points
                query_indices = torch.stack([
                    torch.tensor(np.random.choice(total_points, n_queries, replace=replace_queries))
                    for _ in range(n_functions)
                ], dim=0).to(self.device)
            
            # Gather coordinate pairs for the support set and query set
            example_xs = torch.stack([self.grid[example_indices[i]] for i in range(n_functions)], dim=0)
            query_xs = torch.stack([self.grid[query_indices[i]] for i in range(n_functions)], dim=0)
            
            # Get the selected functions' data
            selected_S = self.S_train[indices]  # shape: [n_functions, grid_size[0], grid_size[1], 1]
            
            # Reshape to flatten spatial dimensions
            selected_S_flat = selected_S.view(n_functions, -1, 1)  # shape: [n_functions, total_points, 1]
            
            # For examples, gather the sampled points
            example_S = torch.stack([
                selected_S_flat[i][example_indices[i]] for i in range(n_functions)
            ], dim=0)
            
            # For queries, gather the sampled points
            query_S = torch.stack([
                selected_S_flat[i][query_indices[i]] for i in range(n_functions)
            ], dim=0)
            
            # Gather the corresponding alpha coefficients
            alpha = self.alpha_train[indices]
            
            # Add verification of output shapes and values
            assert example_xs.shape == (n_functions, n_examples, 2), f"Expected example_xs shape {(n_functions, n_examples, 2)}, got {example_xs.shape}"
            assert example_S.shape == (n_functions, n_examples, 1), f"Expected example_S shape {(n_functions, n_examples, 1)}, got {example_S.shape}"
            assert query_xs.shape == (n_functions, n_queries, 2), f"Expected query_xs shape {(n_functions, n_queries, 2)}, got {query_xs.shape}"
            assert query_S.shape == (n_functions, n_queries, 1), f"Expected query_S shape {(n_functions, n_queries, 1)}, got {query_S.shape}"
            
            return example_xs, example_S, query_xs, query_S, {
                "alpha_train": alpha,
                "S_mean": self.S_mean,
                "S_std": self.S_std,
                "grid_size": self.grid_size
            }

if __name__ == "__main__":
    # Create an instance of the dataset
    dataset = ChladniDataset(n_functions=10, n_examples=100)
    
    # Test the sample method
    print("\nTesting sample method...")
    example_xs, example_S, query_xs, query_S, metadata = dataset.sample()
    
    print("\nSample outputs:")
    print(f"Example points shape: {example_xs.shape}")
    print(f"Example values shape: {example_S.shape}")
    print(f"Query points shape: {query_xs.shape}")
    print(f"Query values shape: {query_S.shape}")
    print(f"Alpha shape: {metadata['alpha_train'].shape}")
    
    # Test a few samples to ensure randomness
    print("\nTesting multiple samples for randomness...")
    for i in range(3):
        _, example_S_new, _, _, _ = dataset.sample()
        print(f"Sample {i+1} mean value: {example_S_new.mean():.3f}")