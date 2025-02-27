import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

# Import the necessary classes
from FunctionEncoder import FunctionEncoder
from ChladniDataset import ChladniDataset  # For S_train
from ChladniDataset_u import ChladniDataset as ChladniDataset_Z  # For Z_train

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=110)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--n_samples", type=int, default=10000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--s_model_path", type=str, default="parameterized_Chladni_EncoderOnly/2025-02-25_13-08-10")
parser.add_argument("--z_model_path", type=str, default="parameterized_Chladni_Z_EncoderOnly/2025-02-25_13-02-10")
parser.add_argument("--output_dir", type=str, default="basis_to_basis_dataset")
args = parser.parse_args()

# Set random seeds for reproducibility
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device:', device)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load datasets
s_dataset = ChladniDataset(n_functions=args.n_samples)
z_dataset = ChladniDataset_Z(n_functions=args.n_samples)

# Verify datasets have the same parameters
assert s_dataset.n_functions == z_dataset.n_functions, "Datasets must have the same number of functions"
print(f"Number of samples: {s_dataset.n_functions}")

# Load S_train model (force basis)
s_model = FunctionEncoder(
    input_size=s_dataset.input_size,
    output_size=s_dataset.output_size,
    data_type=s_dataset.data_type,
    n_basis=args.n_basis,
    model_type='MLP',
    method=args.train_method,
    use_residuals_method=False
).to(device)
s_model.load_state_dict(torch.load(f"{args.s_model_path}/model.pth"))
s_model.eval()
print("Loaded S_train model")

# Load Z_train model (displacement basis)
z_model = FunctionEncoder(
    input_size=z_dataset.input_size,
    output_size=z_dataset.output_size,
    data_type=z_dataset.data_type,
    n_basis=args.n_basis,
    model_type='MLP',
    method=args.train_method,
    use_residuals_method=False
).to(device)
z_model.load_state_dict(torch.load(f"{args.z_model_path}/model.pth"))
z_model.eval()
print("Loaded Z_train model")

# Initialize storage for representations
s_representations = []
z_representations = []

print(f"Computing representations for {args.n_samples} samples...")

# Process each sample and compute representations
with torch.no_grad():
    for i in range(args.n_samples):
        # Get the full grid of coordinate pairs
        grid = s_dataset.grid
        total_points = s_dataset.num_points
        
        # For computing the representation, select a support set
        # Instead of random points, use a stratified sample for better coverage
        n_examples = s_dataset.n_examples
        
        # Create a systematic sampling pattern (can be modified if needed)
        # This ensures better coverage of the domain compared to random sampling
        step = max(1, total_points // n_examples)
        indices = torch.arange(0, total_points, step)[:n_examples]
        
        # For the remaining points (if any), sample randomly
        if len(indices) < n_examples:
            remaining = n_examples - len(indices)
            available = torch.ones(total_points, dtype=torch.bool)
            available[indices] = False
            remaining_indices = torch.randperm(available.sum())[:remaining]
            all_available = torch.where(available)[0]
            additional_indices = all_available[remaining_indices]
            indices = torch.cat([indices, additional_indices])
        
        # Prepare input coordinates
        support_xs = grid[indices].unsqueeze(0).to(device)  # shape: (1, n_examples, 2)
        
        # Get S_train values for the current sample
        s_train_flat = s_dataset.S_train[i].view(-1, 1)  # shape: (total_points, 1)
        support_s = s_train_flat[indices].unsqueeze(0).to(device)  # shape: (1, n_examples, 1)
        
        # Get Z_train values for the current sample
        z_train_flat = z_dataset.Z_train[i].reshape(-1, 1)  # shape: (total_points, 1)
        support_z = z_train_flat[indices].unsqueeze(0).to(device)  # shape: (1, n_examples, 1)
        
        # Compute representations
        s_rep, _ = s_model.compute_representation(support_xs, support_s)
        z_rep, _ = z_model.compute_representation(support_xs, support_z)
        
        # Store representations
        s_representations.append(s_rep.cpu())
        z_representations.append(z_rep.cpu())
        
        # Print progress
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Processed {i + 1}/{args.n_samples} samples")

# Convert lists to tensors
s_representations = torch.stack(s_representations)
z_representations = torch.stack(z_representations)

print(f"S representations shape: {s_representations.shape}")
print(f"Z representations shape: {z_representations.shape}")

# Check if representations have non-zero values
print(f"S representations min: {s_representations.min():.8f}, max: {s_representations.max():.8f}")
print(f"Z representations min: {z_representations.min():.8f}, max: {z_representations.max():.8f}")

# Save datasets
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_file = f"{args.output_dir}/representations_{timestamp}.pt"
torch.save({
    's_representations': s_representations,
    'z_representations': z_representations,
    'n_basis': args.n_basis,
    'n_samples': args.n_samples,
    'seed': seed
}, output_file)

print(f"Saved representations to {output_file}")

# Visualize a few examples of the representations for verification
plt.figure(figsize=(15, 10))

# Plot a few S representations
for i in range(min(5, args.n_samples)):
    plt.subplot(2, 5, i + 1)
    rep_data = s_representations[i, 0].numpy()
    plt.plot(range(len(rep_data)), rep_data)
    plt.title(f"S representation {i+1}")
    plt.grid(True)
    # Add y-axis limits if needed based on data range
    plt.ylim(rep_data.min() - 0.1, rep_data.max() + 0.1)

# Plot a few Z representations
for i in range(min(5, args.n_samples)):
    plt.subplot(2, 5, i + 6)
    rep_data = z_representations[i, 0].numpy()
    plt.plot(range(len(rep_data)), rep_data)
    plt.title(f"Z representation {i+1}")
    plt.grid(True)
    # Add y-axis limits if needed based on data range
    plt.ylim(rep_data.min() - 0.1, rep_data.max() + 0.1)

plt.tight_layout()
plt.savefig(f"{args.output_dir}/representation_examples_{timestamp}.png", dpi=300)
print(f"Saved visualization to {args.output_dir}/representation_examples_{timestamp}.png")

# Also create a scatterplot to show correlation between a few coefficients
plt.figure(figsize=(15, 10))
for idx in range(min(6, args.n_basis)):
    plt.subplot(2, 3, idx + 1)
    plt.scatter(s_representations[:, 0, idx].numpy(), z_representations[:, 0, idx].numpy(), alpha=0.5)
    plt.xlabel(f"S coefficient {idx}")
    plt.ylabel(f"Z coefficient {idx}")
    plt.title(f"Coefficient {idx} correlation")
    plt.grid(True)

plt.tight_layout()
plt.savefig(f"{args.output_dir}/coefficient_correlation_{timestamp}.png", dpi=300)
print(f"Saved correlation plot to {args.output_dir}/coefficient_correlation_{timestamp}.png")
