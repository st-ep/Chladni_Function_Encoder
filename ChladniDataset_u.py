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
                 file_url: str = "https://drive.usercontent.google.com/download?id=1h1dBm6RzJb6YrFgIRYysLquzjHpqZTk5&export=download&authuser=0&confirm=t&uuid=02f95338-1a84-4652-8624-41498c496c86&at=AEz70l4e_neaGVk3dOsdCyjQx1s_:1740507181182",
                 local_file: str = "ChladniData.mat",
                 device: str = "auto",
                 normalize_z: bool = True,  # Added parameter to control normalization
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
            # Print available keys in the file for debugging
            print(f"Available keys in the .mat file: {list(f.keys())}")
            
            alpha_train = np.array(f['alpha_train']).transpose()
            Z_train = np.array(f['Z_train']).transpose()
            
            # Print shape and stats of the loaded data
            print(f"Alpha train shape: {alpha_train.shape}")
            print(f"Z_train shape: {Z_train.shape}")
            print(f"Z_train min/max: {np.min(Z_train):.8f}, {np.max(Z_train):.8f}")
            
            x = np.array(f['x']).squeeze()
            y = np.array(f['y']).squeeze()
        
        # Convert to tensors
        self.alpha_train = torch.tensor(alpha_train, dtype=self.dtype, device=self.device)
        Z_train = torch.tensor(Z_train, dtype=self.dtype, device=self.device)
        
        # Store original statistics for reference
        self.Z_mean = Z_train.mean()
        self.Z_std = Z_train.std()
        print(f"Z_train mean: {self.Z_mean:.8f}, std: {self.Z_std:.8f}")
        
        # Check dimensions and reshape appropriately
        print(f"Z_train tensor shape: {Z_train.shape}, ndim: {Z_train.ndim}")
        
        # From the MATLAB code, Z_train shape should be [25, 25, 10000]
        # We want to reshape to [10000, 25, 25, 1]
        Z_train = Z_train.permute(2, 0, 1)  # Now [10000, 25, 25]
        
        # Scale the Z values to a more typical range for neural networks
        if normalize_z:
            # Find the global min and max across all samples
            Z_min = Z_train.min()
            Z_max = Z_train.max()
            
            # Store scaling factors for later use
            self.Z_min = Z_min
            self.Z_max = Z_max
            
            # Scale to [-1, 1] range if there's a non-zero range
            if Z_max > Z_min:
                print(f"Scaling Z values from [{Z_min:.8f}, {Z_max:.8f}] to [-1, 1]")
                Z_train = 2.0 * (Z_train - Z_min) / (Z_max - Z_min) - 1.0
            else:
                print("Warning: Z values have zero range, skipping normalization")
        
        self.Z_train = Z_train.unsqueeze(-1)  # Now [10000, 25, 25, 1]
        
        # Print final shape and values after processing
        print(f"Final Z_train shape: {self.Z_train.shape}")
        print(f"Z_train value range after processing: [{self.Z_train.min():.6f}, {self.Z_train.max():.6f}]")
        
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
        self.N_train = self.alpha_train.shape[2]  # Changed to match alpha_train shape
        
        # Store grid dimensions for reference
        self.grid_size = (len(x), len(y))
        
        # Add data validation checks after loading
        print(f"Dataset Statistics:")
        print(f"Z_train shape: {self.Z_train.shape}")
        print(f"Grid shape: {self.grid.shape}")
        print(f"Grid dimensions: {self.grid_size[0]}x{self.grid_size[1]} ({self.num_points} total points)")
        
        # Print value ranges
        print("\nValue ranges:")
        print(f"Alpha train: [{self.alpha_train.min():.3f}, {self.alpha_train.max():.3f}]")
        print(f"Z_train: [{self.Z_train.min():.3f}, {self.Z_train.max():.3f}]")
        print(f"Grid x: [{self.grid[:, 0].min():.3f}, {self.grid[:, 0].max():.3f}]")
        print(f"Grid y: [{self.grid[:, 1].min():.3f}, {self.grid[:, 1].max():.3f}]")

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
          - example_xs: (n_functions, n_examples, 2) coordinate pairs for support points.
          - example_Z: (n_functions, n_examples, 1) corresponding displacement values.
          - query_xs: (n_functions, n_queries, 2) coordinate pairs for query points.
          - query_Z: (n_functions, n_queries, 1) corresponding displacement values.
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
            selected_Z = self.Z_train[indices]  # shape: [n_functions, 25, 25, 1]
            
            # Reshape to flatten spatial dimensions - use reshape instead of view
            selected_Z_flat = selected_Z.reshape(n_functions, -1, 1)  # shape: [n_functions, 625, 1]
            
            # For examples, gather the sampled points
            example_Z = torch.stack([
                selected_Z_flat[i][example_indices[i]] for i in range(n_functions)
            ], dim=0)
            
            # For queries, gather the sampled points
            query_Z = torch.stack([
                selected_Z_flat[i][query_indices[i]] for i in range(n_functions)
            ], dim=0)
            
            # Gather the corresponding alpha coefficients
            # Need to reshape alpha_train to get the right indices
            alpha = torch.stack([
                self.alpha_train[:, :, i].reshape(-1) for i in indices
            ], dim=0)
            
            # Add verification of output shapes and values
            assert example_xs.shape == (n_functions, n_examples, 2), f"Expected example_xs shape {(n_functions, n_examples, 2)}, got {example_xs.shape}"
            assert example_Z.shape == (n_functions, n_examples, 1), f"Expected example_Z shape {(n_functions, n_examples, 1)}, got {example_Z.shape}"
            assert query_xs.shape == (n_functions, n_queries, 2), f"Expected query_xs shape {(n_functions, n_queries, 2)}, got {query_xs.shape}"
            assert query_Z.shape == (n_functions, n_queries, 1), f"Expected query_Z shape {(n_functions, n_queries, 1)}, got {query_Z.shape}"
            
            return example_xs, example_Z, query_xs, query_Z, {
                "alpha_train": alpha,
                "Z_mean": self.Z_mean,
                "Z_std": self.Z_std,
                "Z_min": getattr(self, "Z_min", None),  # Include scaling factors in metadata
                "Z_max": getattr(self, "Z_max", None),
                "grid_size": self.grid_size
            }

if __name__ == "__main__":
    # Create an instance of the dataset
    dataset = ChladniDataset(n_functions=10, n_examples=100)
    
    # Test the sample method
    print("\nTesting sample method...")
    example_xs, example_Z, query_xs, query_Z, metadata = dataset.sample()
    
    print("\nSample outputs:")
    print(f"Example points shape: {example_xs.shape}")
    print(f"Example values shape: {example_Z.shape}")
    print(f"Query points shape: {query_xs.shape}")
    print(f"Query values shape: {query_Z.shape}")
    print(f"Alpha shape: {metadata['alpha_train'].shape}")
    
    # Print value statistics with higher precision
    print(f"Example Z values - min: {example_Z.min():.6f}, max: {example_Z.max():.6f}, mean: {example_Z.mean():.6f}")
    print(f"Query Z values - min: {query_Z.min():.6f}, max: {query_Z.max():.6f}, mean: {query_Z.mean():.6f}")
    
    # Test a few samples to ensure randomness
    print("\nTesting multiple samples for randomness...")
    for i in range(3):
        _, example_Z_new, _, _, _ = dataset.sample()
        print(f"Sample {i+1} mean value: {example_Z_new.mean():.6f}")