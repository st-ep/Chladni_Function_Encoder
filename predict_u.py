import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import h5py

from FunctionEncoder import FunctionEncoder
from ChladniDataset import ChladniDataset  # For S_train
from ChladniDataset_u import ChladniDataset as ChladniDataset_Z  # For Z_train

# Define the B2B network architecture (match exactly with b2b.py)
class BasisToBasicsNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 512, 256]):
        super(BasisToBasicsNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=110)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--n_samples", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--s_model_path", type=str, default="parameterized_Chladni_EncoderOnly/2025-02-25_13-08-10")
parser.add_argument("--z_model_path", type=str, default="parameterized_Chladni_Z_EncoderOnly/2025-02-25_13-02-10")
parser.add_argument("--b2b_model_path", type=str, default="b2b_model_results/best_model.pt")
parser.add_argument("--output_dir", type=str, default="displacement_predictions")
parser.add_argument("--evaluate_test", action="store_true", help="Evaluate on test set instead of training set")
args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load datasets
s_dataset = ChladniDataset(n_functions=args.n_samples)
z_dataset = ChladniDataset_Z(n_functions=args.n_samples)

print(f"Dataset contains {s_dataset.n_functions} samples")

# Load models
# 1. Load the S model (force basis)
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
print("Loaded S (force) model")

# 2. Load the Z model (displacement basis)
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
print("Loaded Z (displacement) model")

# 3. Load the B2B mapping model
b2b_checkpoint = torch.load(args.b2b_model_path)
input_dim = b2b_checkpoint['input_dim']
output_dim = b2b_checkpoint['output_dim']
hidden_dims = b2b_checkpoint['hidden_dims']

# Create the model with the exact same architecture as in b2b.py
b2b_model = BasisToBasicsNetwork(input_dim, output_dim, hidden_dims).to(device)
print(f"Created B2B model with architecture matching the saved checkpoint")
b2b_model.load_state_dict(b2b_checkpoint['model_state_dict'])
b2b_model.eval()
print("Loaded B2B (force-to-displacement mapping) model")

# Evaluate on train or test samples
total_mse = 0.0
max_plots = 10  # Maximum number of samples to visualize

# Determine which dataset to use for evaluation
if args.evaluate_test:
    print("Evaluating on test dataset...")
    # Access the test data from the dataset
    local_file = "ChladniData.mat"
    
    with h5py.File(local_file, 'r') as f:
        S_test = np.array(f['S_test']).transpose()
    
    # Convert to tensor and reshape
    S_test = torch.tensor(S_test, dtype=s_dataset.dtype, device=s_dataset.device)
    
    # Print statistics
    print(f"S_test mean: {S_test.mean():.6f}, std: {S_test.std():.6f}")
    print(f"S_train mean: {s_dataset.S_mean:.6f}, std: {s_dataset.S_std:.6f}")
    
    # Process S_test same as S_train
    S_test = S_test.permute(2, 0, 1)
    S_test = S_test.contiguous()
    S_test = S_test.unsqueeze(-1)
    
    # Get Z test data
    with h5py.File(local_file, 'r') as f:
        Z_test = np.array(f['Z_test']).transpose()
    
    # Convert to tensor
    Z_test = torch.tensor(Z_test, dtype=z_dataset.dtype, device=z_dataset.device)
    
    # Print original min/max (not just mean/std)
    Z_test_min = Z_test.min()
    Z_test_max = Z_test.max()
    print(f"Z_test original min: {Z_test_min:.8f}, max: {Z_test_max:.8f}")
    
    # Process Z_test same as Z_train
    Z_test = Z_test.permute(2, 0, 1)  # Now [n_samples, 25, 25]
    Z_test = Z_test.contiguous()
    Z_test = Z_test.unsqueeze(-1)  # Now [n_samples, 25, 25, 1]
    
    # Apply the SAME normalization as was used for training data
    Z_train_min = z_dataset.Z_min
    Z_train_max = z_dataset.Z_max
    print(f"Z_train normalization range: [{Z_train_min:.8f}, {Z_train_max:.8f}]")
    print(f"Z_test original range: [{Z_test_min:.8f}, {Z_test_max:.8f}]")

    Z_train_range = Z_train_max - Z_train_min
    if Z_train_range > 0:  # Avoid division by zero
        Z_test = 2 * (Z_test - Z_train_min) / Z_train_range - 1
    
    print(f"Z_test after scaling with training parameters - min: {Z_test.min():.6f}, max: {Z_test.max():.6f}")
    
    # Set up evaluation parameters
    total_samples = min(args.n_samples, S_test.shape[0])
    print(f"Evaluating on {total_samples} test samples")
    
    # Create output directory
    test_output_dir = os.path.join(args.output_dir, "test_results")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Use processed data
    S_eval = S_test
    Z_eval = Z_test
    output_dir = test_output_dir
    
    # Store min/max for denormalization
    Z_min_val = Z_test_min.item()
    Z_max_val = Z_test_max.item() 
else:
    print("Evaluating on training dataset...")
    total_samples = min(args.n_samples, len(s_dataset.S_train))
    S_eval = s_dataset.S_train
    Z_eval = z_dataset.Z_train
    output_dir = args.output_dir
    
    # For training data, we already know the original min/max from dataset logs
    Z_min_val = -0.00009706  # From dataset log
    Z_max_val = 0.00009892   # From dataset log

# Store scaling parameters for consistent denormalization
Z_range = Z_max_val - Z_min_val

with torch.no_grad():
    for sample_idx in range(total_samples):
        # Process grid and support set
        grid = s_dataset.grid
        total_points = s_dataset.num_points
        n_examples = s_dataset.n_examples
        indices = torch.randperm(total_points)[:n_examples]
        support_xs = grid[indices].unsqueeze(0).to(device)
        
        # Get S values for current sample
        s_eval_flat = S_eval[sample_idx].reshape(-1, 1)
        support_s = s_eval_flat[indices].unsqueeze(0).to(device)
        
        # 3-step pipeline
        s_rep, _ = s_model.compute_representation(support_xs, support_s)
        s_rep_flat = s_rep.view(1, -1)
        z_rep_flat = b2b_model(s_rep_flat)
        z_rep = z_rep_flat.view(s_rep.shape)
        grid_batch = grid.unsqueeze(0).to(device)
        Z_pred = z_model.predict(grid_batch, z_rep)
        
        # Get ground truth
        Z_true = Z_eval[sample_idx].to(device)
        
        # Calculate MSE
        sample_mse = ((Z_pred[0].cpu() - Z_true.cpu().reshape(-1, 1))**2).mean().item()
        total_mse += sample_mse
        
        # Print diagnostics for first few samples
        if sample_idx < 3:
            print(f"\nSample {sample_idx+1} diagnostics:")
            print(f"  Z_pred min: {Z_pred.min():.6f}, max: {Z_pred.max():.6f}")
            print(f"  Z_true min: {Z_true.min():.6f}, max: {Z_true.max():.6f}")
        
        # Only print MSE for the first max_plots samples
        if sample_idx < max_plots:
            print(f"Sample {sample_idx+1} MSE: {sample_mse:.6f}")
        
        # Only create visualizations for the first max_plots samples
        if sample_idx < max_plots:
            # Reshape for plotting
            num_side = int(np.sqrt(total_points))
            
            # Ensure all tensors are on CPU before operations
            Z_pred_cpu = Z_pred[0].cpu()
            Z_true_cpu = Z_true.cpu().reshape(-1, 1)
            
            # NO DENORMALIZATION - use the normalized [-1, 1] values directly
            
            # Log scale information
            if sample_idx == 0:
                print(f"\nPlotting diagnostics:")
                print(f"  Z_pred (normalized) - min: {Z_pred_cpu.min():.6f}, max: {Z_pred_cpu.max():.6f}")
                print(f"  Z_true (normalized) - min: {Z_true_cpu.min():.6f}, max: {Z_true_cpu.max():.6f}")
            
            # Reshape to images - use normalized values
            Z_pred_img = Z_pred_cpu.reshape(num_side, num_side).numpy()
            Z_true_img = Z_true_cpu.reshape(num_side, num_side).numpy()
            
            # 1. Displacement field comparison
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # Use same color scale for both plots
            vmin = min(Z_pred_img.min(), Z_true_img.min())
            vmax = max(Z_pred_img.max(), Z_true_img.max())
            
            im = axs[0].contourf(Z_pred_img, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
            axs[0].set_title('Predicted Z (Normalized)')
            fig.colorbar(im, ax=axs[0])
            
            im = axs[1].contourf(Z_true_img, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
            axs[1].set_title('True Z (Normalized)')
            fig.colorbar(im, ax=axs[1])
            
            im = axs[2].contourf(np.abs(Z_true_img - Z_pred_img), levels=50, cmap='viridis')
            axs[2].set_title('Absolute Error')
            fig.colorbar(im, ax=axs[2])
            
            plt.suptitle(f'Sample {sample_idx + 1} - MSE: {sample_mse:.6f}')
            fig.savefig(f'{output_dir}/displacement_{sample_idx + 1}.png', dpi=400, bbox_inches='tight')
            plt.close(fig)
            
            # 2. Chladni pattern comparison (zero contour lines)
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
            
            # True Chladni pattern
            axs[0].contour(Z_true_img, levels=[0], colors=['#d9a521'], linewidths=2)
            axs[0].set_title('True Chladni Pattern', color='white')
            axs[0].set_facecolor('black')
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            
            # Predicted Chladni pattern
            axs[1].contour(Z_pred_img, levels=[0], colors=['#d9a521'], linewidths=2)
            axs[1].set_title('Predicted Chladni Pattern', color='white')
            axs[1].set_facecolor('black')
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            
            plt.tight_layout()
            fig.savefig(f'{output_dir}/chladni_pattern_{sample_idx + 1}.png', dpi=400, bbox_inches='tight')
            plt.close(fig)

# Calculate and report average MSE
avg_mse = total_mse / total_samples
dataset_type = "test" if args.evaluate_test else "training"
print(f"\nAverage MSE across {total_samples} {dataset_type} samples: {avg_mse:.6f}")

# Create a plot summarizing the pipeline
plt.figure(figsize=(12, 8))
plt.text(0.5, 0.9, "Force-to-Displacement Prediction Pipeline", 
         horizontalalignment='center', fontsize=16, fontweight='bold')

plt.text(0.5, 0.8, f"Average MSE on {dataset_type} set: {avg_mse:.6f}", 
         horizontalalignment='center', fontsize=14)

plt.text(0.1, 0.6, "Step 1: Encode Force (S) to S-basis", fontsize=12, fontweight='bold')
plt.text(0.1, 0.55, "Using FunctionEncoder from parameterized_Chladni_EncoderOnly", fontsize=10)

plt.text(0.1, 0.45, "Step 2: Map S-basis to Z-basis", fontsize=12, fontweight='bold')
plt.text(0.1, 0.4, "Using B2B neural network model", fontsize=10)

plt.text(0.1, 0.3, "Step 3: Decode Z-basis to Displacement (Z)", fontsize=12, fontweight='bold')
plt.text(0.1, 0.25, "Using FunctionEncoder from parameterized_Chladni_Z_EncoderOnly", fontsize=10)

plt.axis('off')
plt.savefig(f'{output_dir}/pipeline_summary.png', dpi=400, bbox_inches='tight')
plt.close()

print(f"Results saved to {output_dir}")
