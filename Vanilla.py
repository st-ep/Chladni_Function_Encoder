import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import h5py
import deepxde as dde
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ChladniDataset import ChladniDataset  # For S_train
from ChladniDataset_u import ChladniDataset as ChladniDataset_Z  # For Z_train

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--branch_layers", type=str, default="512,512,256", 
                    help="Comma-separated list of branch network layer dimensions")
parser.add_argument("--trunk_layers", type=str, default="512,512,256", 
                    help="Comma-separated list of trunk network layer dimensions")
parser.add_argument("--output_dim", type=int, default=256, 
                    help="Final output dimension for both branch and trunk networks")
parser.add_argument("--epochs", type=int, default=150000)
parser.add_argument("--batch_size", type=int, default=None, 
                    help="Batch size for training, None for full batch")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--n_samples_train", type=int, default=-1, 
                    help="Number of samples for training, use -1 for all available samples")
parser.add_argument("--n_samples_test", type=int, default=200,
                    help="Number of samples for testing")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="vanilla_deeponet_results")
parser.add_argument("--data_file", type=str, default="ChladniData.mat",
                    help="Path to ChladniData.mat file")
parser.add_argument("--use_provided_test", action="store_true", 
                    help="Use test data directly from .mat file instead of splitting training data")
args = parser.parse_args()

# Set random seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Parse network layer dimensions
branch_layers = [int(dim) for dim in args.branch_layers.split(',')]
trunk_layers = [int(dim) for dim in args.trunk_layers.split(',')]

# Make sure both networks end with the same output dimension
if branch_layers[-1] != args.output_dim:
    branch_layers[-1] = args.output_dim  # Set the final dimension to match
if trunk_layers[-1] != args.output_dim:
    trunk_layers[-1] = args.output_dim  # Set the final dimension to match

print(f"""
Hyperparameters:
- Epochs: {args.epochs}
- Learning Rate: {args.learning_rate}
- Branch Network: {branch_layers}
- Trunk Network: {trunk_layers}
- Network Final Output Dim: {args.output_dim}
- Batch Size: {args.batch_size}
- Training Samples: {args.n_samples_train}
- Testing Samples: {args.n_samples_test}
""")

# Load datasets
s_dataset = ChladniDataset(n_functions=args.n_samples_train + args.n_samples_test)
z_dataset = ChladniDataset_Z(n_functions=args.n_samples_train + args.n_samples_test)

print(f"Dataset contains {s_dataset.n_functions} samples")
print(f"S shape: {s_dataset.S_train.shape}")
print(f"Z shape: {z_dataset.Z_train.shape}")
print(f"Grid shape: {s_dataset.grid.shape}")

# Convert to numpy for deepxde
def prepare_data_for_deeponet():
    """Prepare data in the format required by DeepONet"""
    # Get the grid and force/displacement data
    grid = s_dataset.grid.cpu().numpy()
    
    if args.use_provided_test:
        print("Using test data directly from .mat file...")
        # Load test data from .mat file
        with h5py.File(args.data_file, 'r') as f:
            S_test = np.array(f['S_test']).transpose()
            Z_test = np.array(f['Z_test']).transpose()
        
        # Convert to tensors with same processing as training data
        S_test = torch.tensor(S_test, dtype=s_dataset.dtype, device=s_dataset.device)
        Z_test = torch.tensor(Z_test, dtype=z_dataset.dtype, device=z_dataset.device)
        
        # Process S_test same as S_train
        S_test = S_test.permute(2, 0, 1)
        S_test = S_test.contiguous()
        S_test = S_test.unsqueeze(-1)
        
        # Process Z_test same as Z_train
        Z_test = Z_test.permute(2, 0, 1)
        Z_test = Z_test.contiguous()
        Z_test = Z_test.unsqueeze(-1)
        
        # Apply normalization to Z_test using Z_train parameters (if your dataset applies normalization)
        if hasattr(z_dataset, 'Z_min') and hasattr(z_dataset, 'Z_max'):
            Z_min, Z_max = z_dataset.Z_min, z_dataset.Z_max
            Z_range = Z_max - Z_min
            if Z_range > 0:  # Avoid division by zero
                Z_test = 2 * (Z_test - Z_min) / Z_range - 1
                print(f"Normalized Z_test using Z_train parameters: min={Z_min:.8f}, max={Z_max:.8f}")
        
        # Get training data
        s_data = s_dataset.S_train.cpu().numpy()
        z_data = z_dataset.Z_train.cpu().numpy()
        
        # Get testing data
        s_test_data = S_test.cpu().numpy()
        z_test_data = Z_test.cpu().numpy()
        
        print(f"Training data shapes - S: {s_data.shape}, Z: {z_data.shape}")
        print(f"Test data shapes - S: {s_test_data.shape}, Z: {z_test_data.shape}")
        
        # Use all available training samples if n_samples_train is -1
        n_train = len(s_data) if args.n_samples_train == -1 else min(args.n_samples_train, len(s_data))
        n_test = min(args.n_samples_test, s_test_data.shape[0])
    else:
        # Original code path - split training data into train/test
        print("Splitting training data into train/test sets...")
        s_data = s_dataset.S_train.cpu().numpy()
        z_data = z_dataset.Z_train.cpu().numpy()
        
        # Use all available samples for training minus test samples if n_samples_train is -1
        total_available = len(s_data)
        if args.n_samples_train == -1:
            n_train = max(total_available - args.n_samples_test, 0)
        else:
            n_train = min(args.n_samples_train, total_available - args.n_samples_test)
        n_test = min(args.n_samples_test, total_available - n_train)
    
    # Reshape data
    n_side = int(np.sqrt(s_dataset.num_points))
    
    # Prepare branch input (forcing function on grid)
    branch_inputs_train = []
    for i in range(n_train):
        # Reshape to remove the channel dimension and flatten
        branch_inputs_train.append(s_data[i].reshape(n_side, n_side).flatten())
    branch_inputs_train = np.array(branch_inputs_train)
    
    # Prepare branch input for test data
    branch_inputs_test = []
    if args.use_provided_test:
        for i in range(n_test):
            branch_inputs_test.append(s_test_data[i].reshape(n_side, n_side).flatten())
    else:
        for i in range(n_test):
            branch_inputs_test.append(s_data[n_train + i].reshape(n_side, n_side).flatten())
    branch_inputs_test = np.array(branch_inputs_test)
    
    # Prepare trunk input (coordinates)
    trunk_inputs = grid
    
    # Prepare outputs for training
    outputs_train = []
    for i in range(n_train):
        # Reshape the displacement field
        outputs_train.append(z_data[i].reshape(n_side, n_side).flatten())
    outputs_train = np.array(outputs_train)
    
    # Prepare outputs for testing
    outputs_test = []
    if args.use_provided_test:
        for i in range(n_test):
            outputs_test.append(z_test_data[i].reshape(n_side, n_side).flatten())
    else:
        for i in range(n_test):
            outputs_test.append(z_data[n_train + i].reshape(n_side, n_side).flatten())
    outputs_test = np.array(outputs_test)
    
    print(f"Branch input train shape: {branch_inputs_train.shape}")
    print(f"Branch input test shape: {branch_inputs_test.shape}")
    print(f"Trunk input shape: {trunk_inputs.shape}")
    print(f"Outputs train shape: {outputs_train.shape}")
    print(f"Outputs test shape: {outputs_test.shape}")
    
    # Return all data
    s_all_data = s_data if not args.use_provided_test else np.concatenate([s_data[:n_train], s_test_data[:n_test]], axis=0)
    
    return (branch_inputs_train, trunk_inputs), outputs_train, (branch_inputs_test, trunk_inputs), outputs_test, s_all_data

# Prepare data
x_train, y_train, x_test, y_test, s_data_all = prepare_data_for_deeponet()

# Create DeepONet Data
data = dde.data.Triple(x_train, y_train, x_test, y_test)

# Create DeepONet model
n_side = int(np.sqrt(s_dataset.num_points))
branch_net_size = [n_side * n_side] + branch_layers  # Input size is the flattened grid
trunk_net_size = [2] + trunk_layers  # Input size is 2D coordinates

print(f"Branch network architecture: {branch_net_size}")
print(f"Trunk network architecture: {trunk_net_size}")
print(f"Both networks output dimension: {args.output_dim}")

# Note: Fixed constructor to match the DeepONet API in deepxde
net = dde.maps.DeepONetCartesianProd(
    branch_net_size,
    trunk_net_size,
    "relu",
    "Glorot normal"
)

# Create model
model = dde.Model(data, net)

# Compile model
model.compile(
    "adam",
    lr=args.learning_rate,
    decay=("inverse time", 1, 1e-4),
    metrics=["mean l2 relative error"],
)

# Define model checkpoint path and ensure it's a directory
model_checkpoint_dir = os.path.join(args.output_dir, "model_checkpoints")
os.makedirs(model_checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(model_checkpoint_dir, "best_model")

# Change the checkpoint callback to save less frequently and be less verbose
checkpoint_callback = dde.callbacks.ModelCheckpoint(
    best_model_path,
    verbose=0,  # Reduced verbosity 
    save_better_only=True,
    period=5000  # Only save every 5000 iterations instead of 1000
)

# Add a custom progress display callback
class ProgressCallback(dde.callbacks.Callback):
    def __init__(self, total_iterations, frequency=1000):
        super().__init__()
        self.frequency = frequency
        self.prev_time = None
        self.iter_count = 0
        self.best_loss = float('inf')
        self.total_iterations = total_iterations
        
    def on_train_begin(self):
        print("\nTraining started...")
        self.prev_time = datetime.now()
        # total_iterations is now set in the constructor
        
    def on_epoch_end(self, **kwargs):
        """Called at the end of each epoch"""
        self.iter_count += 1
        
        # Only print at specified frequency
        if self.iter_count % self.frequency != 0 and self.iter_count != 1:
            return
            
        loss_train = kwargs.get("loss_train", None)
        loss_test = kwargs.get("loss_test", None)
        metrics_test = kwargs.get("metrics_test", None)
        
        # Calculate time elapsed
        current_time = datetime.now()
        duration = (current_time - self.prev_time).total_seconds()
        self.prev_time = current_time
        
        # Calculate progress percentage
        progress = min(100, int(100 * self.iter_count / self.total_iterations))
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Get string representation of losses and metrics
        loss_train_str = f"{loss_train:.2e}" if loss_train is not None else "-"
        loss_test_str = f"{loss_test:.2e}" if loss_test is not None else "-"
        
        metric_str = "-"
        if metrics_test is not None and len(metrics_test) > 0:
            metric_str = f"{metrics_test[0]:.2e}"
        
        # Clear the current line and print the progress bar
        print(f"\r[{bar}] {progress}% | Iter: {self.iter_count}/{self.total_iterations} | Train: {loss_train_str} | Test: {loss_test_str} | Metric: {metric_str}", end="")
        
        # Track best model - print on a new line
        if loss_test is not None and loss_test < self.best_loss:
            self.best_loss = loss_test
            print(f"\n→ New best model! Test loss: {loss_test:.2e}")
    
    def on_train_end(self):
        print("\nTraining completed. Best test loss: {:.2e}".format(self.best_loss))
        print("=" * 80)

# Create a progress callback - pass the iterations to the callback
progress_callback = ProgressCallback(total_iterations=args.epochs, frequency=1000)  # Show progress every 1000 iterations

# Create a cleaner directory structure for checkpoints
# Use iterations instead of epochs since epochs is deprecated
iterations = args.epochs

# Only keep the best model, no additional checkpoints 
losshistory, train_state = model.train(
    iterations=iterations, 
    batch_size=args.batch_size,
    display_every=1000,  # Change from 0 to 1000 to avoid division by zero
    callbacks=[checkpoint_callback, progress_callback],
    disregard_previous_best=True,  # Start fresh with this training run
    model_restore_path=None  # Don't try to restore a previous model
)

# Custom functions to plot loss and metrics history
def plot_loss_history(loss_history):
    """Custom function to plot the loss history"""
    plt.figure(figsize=(12, 8))
    steps = loss_history.steps
    loss_train = loss_history.loss_train
    loss_test = loss_history.loss_test
    
    plt.semilogy(steps, loss_train, label="Train loss")
    plt.semilogy(steps, loss_test, label="Test loss")
    
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=14)
    plt.title("Training History", fontsize=18)
    return plt

def plot_metrics_history(train_state):
    """Custom function to plot the metrics history"""
    plt.figure(figsize=(12, 8))
    steps = train_state.steps
    
    if hasattr(train_state, 'metrics_test'):
        metrics = train_state.metrics_test
        if metrics and len(metrics) > 0:
            plt.semilogy(steps, metrics, 'b-', label="Test metric (Mean L2 relative error)")
            
            plt.xlabel("Iterations", fontsize=16)
            plt.ylabel("Metric Value", fontsize=16)
            plt.grid(True, which="both", ls="--")
            plt.legend(fontsize=14)
            plt.title("Metrics History", fontsize=18)
        else:
            plt.text(0.5, 0.5, "No metrics data available", 
                     horizontalalignment='center', fontsize=16)
    else:
        plt.text(0.5, 0.5, "No metrics_test attribute in train_state", 
                 horizontalalignment='center', fontsize=16)
    
    return plt

# Create a plots directory
plots_dir = os.path.join(args.output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Plot our custom loss history
try:
    plt_loss = plot_loss_history(losshistory)
    plt_loss.savefig(f"{plots_dir}/loss_history.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Successfully created loss history plot")
except Exception as e:
    print(f"Warning: Could not plot loss history: {str(e)}")

# Plot our custom metrics history
try:
    plt_metrics = plot_metrics_history(train_state)
    plt_metrics.savefig(f"{plots_dir}/metrics_history.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Successfully created metrics history plot")
except Exception as e:
    print(f"Warning: Could not plot metrics history: {str(e)}")

# Do NOT try to use DeepXDE's built-in plotting functions as they don't support DeepONet well
print("Skipping DeepXDE's built-in plotting functions as they don't support multi-input networks like DeepONet")

# No need to explicitly load the model after training - just use the final trained model
# The model is already trained, no need to restore it unless you want a specific checkpoint
print("Using the trained model for prediction without restoring from checkpoint...")

# Test on a few samples
n_test_display = min(4, args.n_samples_test)
grid_np = s_dataset.grid.cpu().numpy()
n_side = int(np.sqrt(s_dataset.num_points))

# Create a results directory for visualizations
results_dir = os.path.join(args.output_dir, "test_results")
os.makedirs(results_dir, exist_ok=True)

# Create a directory for training results
train_results_dir = os.path.join(args.output_dir, "train_results")
os.makedirs(train_results_dir, exist_ok=True)

# Create a summary file for test results
with open(os.path.join(results_dir, "test_summary.txt"), "w") as f:
    f.write(f"DeepONet Test Results\n")
    f.write(f"=====================\n\n")
    f.write(f"Model Architecture:\n")
    f.write(f"  Branch Network: {branch_net_size}\n")
    f.write(f"  Trunk Network: {trunk_net_size}\n\n")

# Create a summary file for training results
with open(os.path.join(train_results_dir, "train_summary.txt"), "w") as f:
    f.write(f"DeepONet Training Results\n")
    f.write(f"=======================\n\n")
    f.write(f"Model Architecture:\n")
    f.write(f"  Branch Network: {branch_net_size}\n")
    f.write(f"  Trunk Network: {trunk_net_size}\n\n")

# Visualize a few training samples
n_train_display = min(4, args.n_samples_train)
print(f"Visualizing {n_train_display} training samples...")

for i in range(n_train_display):
    print(f"Visualizing training sample {i+1}/{n_train_display}")
    
    # Get training data
    train_branch_input = x_train[0][i:i+1]  # Get one sample
    s_input_img = s_data_all[i].reshape(n_side, n_side)
    
    # Predict using DeepONet
    y_pred = model.predict((train_branch_input, grid_np))
    
    # Get ground truth
    y_true = y_train[i]
    
    # Reshape for plotting
    y_pred_img = y_pred.reshape(n_side, n_side)
    y_true_img = y_true.reshape(n_side, n_side)
    
    # Calculate error
    mse = np.mean((y_pred - y_true)**2)
    print(f"Training sample {i+1} MSE: {mse:.6f}")
    
    # Save result to summary file
    with open(os.path.join(train_results_dir, "train_summary.txt"), "a") as f:
        f.write(f"Sample {i+1} MSE: {mse:.6f}\n")
    
    # 0. Input force field
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.contourf(s_input_img, levels=50, cmap='viridis')
    ax.set_title('Input Force Field (S)', fontsize=16)
    ax.set_xlabel('X coordinate', fontsize=14)
    ax.set_ylabel('Y coordinate', fontsize=14)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(f'{train_results_dir}/input_force_{i + 1}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    # 1. Displacement field comparison
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Use same color scale for both plots
    vmin = min(y_pred_img.min(), y_true_img.min())
    vmax = max(y_pred_img.max(), y_true_img.max())
    
    im = axs[0].contourf(y_pred_img, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('Predicted Z (DeepONet)', fontsize=16)
    axs[0].set_xlabel('X coordinate', fontsize=14)
    axs[0].set_ylabel('Y coordinate', fontsize=14)
    fig.colorbar(im, ax=axs[0])
    
    im = axs[1].contourf(y_true_img, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('True Z', fontsize=16)
    axs[1].set_xlabel('X coordinate', fontsize=14)
    axs[1].set_ylabel('Y coordinate', fontsize=14)
    fig.colorbar(im, ax=axs[1])
    
    abs_err = np.abs(y_true_img - y_pred_img)
    im = axs[2].contourf(abs_err, levels=50, cmap='viridis')
    axs[2].set_title('Absolute Error', fontsize=16)
    axs[2].set_xlabel('X coordinate', fontsize=14)
    axs[2].set_ylabel('Y coordinate', fontsize=14)
    fig.colorbar(im, ax=axs[2])
    
    plt.suptitle(f'Training Sample {i + 1} - MSE: {mse:.6f}', fontsize=18)
    plt.tight_layout()
    fig.savefig(f'{train_results_dir}/displacement_{i + 1}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Chladni pattern comparison (zero contour lines)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
    
    # True Chladni pattern
    try:
        cs = axs[0].contour(y_true_img, levels=[0], colors=['#d9a521'], linewidths=2)
        axs[0].set_title('True Chladni Pattern', color='white', fontsize=16)
        axs[0].set_facecolor('black')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
    except Exception as e:
        print(f"Warning: Could not plot true Chladni pattern contour: {str(e)}")
        axs[0].text(0.5, 0.5, 'No zero contour found', 
                    horizontalalignment='center', 
                    color='white', fontsize=14)
    
    # Predicted Chladni pattern
    try:
        cs = axs[1].contour(y_pred_img, levels=[0], colors=['#d9a521'], linewidths=2)
        axs[1].set_title('Predicted Chladni Pattern (DeepONet)', color='white', fontsize=16)
        axs[1].set_facecolor('black')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
    except Exception as e:
        print(f"Warning: Could not plot predicted Chladni pattern contour: {str(e)}")
        axs[1].text(0.5, 0.5, 'No zero contour found', 
                    horizontalalignment='center', 
                    color='white', fontsize=14)
    
    plt.tight_layout()
    fig.savefig(f'{train_results_dir}/chladni_pattern_{i + 1}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Cross-section comparison
    # Take a slice through the middle of the displacement field
    mid_idx = n_side // 2
    y_pred_cross = y_pred_img[mid_idx, :]
    y_true_cross = y_true_img[mid_idx, :]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(n_side), y_true_cross, 'b-', linewidth=2, label='True Z')
    ax.plot(np.arange(n_side), y_pred_cross, 'r--', linewidth=2, label='Predicted Z')
    ax.set_title(f'Cross-section Comparison (Y = {mid_idx})', fontsize=16)
    ax.set_xlabel('X coordinate', fontsize=14)
    ax.set_ylabel('Displacement (Z)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(f'{train_results_dir}/cross_section_{i + 1}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

# Calculate average training error
train_preds = model.predict(x_train)
train_errors = np.mean((train_preds - y_train)**2, axis=1)
avg_train_mse = np.mean(train_errors)
print(f"\nAverage MSE across all training samples: {avg_train_mse:.6f}")

# Save average training MSE to summary file
with open(os.path.join(train_results_dir, "train_summary.txt"), "a") as f:
    f.write(f"\nAverage MSE across all {args.n_samples_train} training samples: {avg_train_mse:.6f}\n")

# Now continue with test results
for i in range(n_test_display):
    print(f"Evaluating test sample {i+1}/{n_test_display}")
    
    # Get test data
    test_branch_input = x_test[0][i:i+1]  # Get one sample
    s_input_img = s_data_all[args.n_samples_train + i].reshape(n_side, n_side)
    
    # Predict using DeepONet
    y_pred = model.predict((test_branch_input, grid_np))
    
    # Get ground truth
    y_true = y_test[i]
    
    # Reshape for plotting
    y_pred_img = y_pred.reshape(n_side, n_side)
    y_true_img = y_true.reshape(n_side, n_side)
    
    # Calculate error
    mse = np.mean((y_pred - y_true)**2)
    print(f"Test sample {i+1} MSE: {mse:.6f}")
    
    # Save result to summary file
    with open(os.path.join(results_dir, "test_summary.txt"), "a") as f:
        f.write(f"Sample {i+1} MSE: {mse:.6f}\n")
    
    # 0. Input force field
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.contourf(s_input_img, levels=50, cmap='viridis')
    ax.set_title('Input Force Field (S)', fontsize=16)
    ax.set_xlabel('X coordinate', fontsize=14)
    ax.set_ylabel('Y coordinate', fontsize=14)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(f'{results_dir}/input_force_{i + 1}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    # 1. Displacement field comparison
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Use same color scale for both plots
    vmin = min(y_pred_img.min(), y_true_img.min())
    vmax = max(y_pred_img.max(), y_true_img.max())
    
    im = axs[0].contourf(y_pred_img, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('Predicted Z (DeepONet)', fontsize=16)
    axs[0].set_xlabel('X coordinate', fontsize=14)
    axs[0].set_ylabel('Y coordinate', fontsize=14)
    fig.colorbar(im, ax=axs[0])
    
    im = axs[1].contourf(y_true_img, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('True Z', fontsize=16)
    axs[1].set_xlabel('X coordinate', fontsize=14)
    axs[1].set_ylabel('Y coordinate', fontsize=14)
    fig.colorbar(im, ax=axs[1])
    
    abs_err = np.abs(y_true_img - y_pred_img)
    im = axs[2].contourf(abs_err, levels=50, cmap='viridis')
    axs[2].set_title('Absolute Error', fontsize=16)
    axs[2].set_xlabel('X coordinate', fontsize=14)
    axs[2].set_ylabel('Y coordinate', fontsize=14)
    fig.colorbar(im, ax=axs[2])
    
    plt.suptitle(f'Test Sample {i + 1} - MSE: {mse:.6f}', fontsize=18)
    plt.tight_layout()
    fig.savefig(f'{results_dir}/displacement_{i + 1}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Chladni pattern comparison (zero contour lines)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
    
    # True Chladni pattern
    try:
        cs = axs[0].contour(y_true_img, levels=[0], colors=['#d9a521'], linewidths=2)
        axs[0].set_title('True Chladni Pattern', color='white', fontsize=16)
        axs[0].set_facecolor('black')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
    except Exception as e:
        print(f"Warning: Could not plot true Chladni pattern contour: {str(e)}")
        axs[0].text(0.5, 0.5, 'No zero contour found', 
                    horizontalalignment='center', 
                    color='white', fontsize=14)
    
    # Predicted Chladni pattern
    try:
        cs = axs[1].contour(y_pred_img, levels=[0], colors=['#d9a521'], linewidths=2)
        axs[1].set_title('Predicted Chladni Pattern (DeepONet)', color='white', fontsize=16)
        axs[1].set_facecolor('black')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
    except Exception as e:
        print(f"Warning: Could not plot predicted Chladni pattern contour: {str(e)}")
        axs[1].text(0.5, 0.5, 'No zero contour found', 
                    horizontalalignment='center', 
                    color='white', fontsize=14)
    
    plt.tight_layout()
    fig.savefig(f'{results_dir}/chladni_pattern_{i + 1}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Cross-section comparison
    # Take a slice through the middle of the displacement field
    mid_idx = n_side // 2
    y_pred_cross = y_pred_img[mid_idx, :]
    y_true_cross = y_true_img[mid_idx, :]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(n_side), y_true_cross, 'b-', linewidth=2, label='True Z')
    ax.plot(np.arange(n_side), y_pred_cross, 'r--', linewidth=2, label='Predicted Z')
    ax.set_title(f'Cross-section Comparison (Y = {mid_idx})', fontsize=16)
    ax.set_xlabel('X coordinate', fontsize=14)
    ax.set_ylabel('Displacement (Z)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(f'{results_dir}/cross_section_{i + 1}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

# Calculate average test error
all_preds = model.predict(x_test)
all_errors = np.mean((all_preds - y_test)**2, axis=1)
avg_mse = np.mean(all_errors)
print(f"\nAverage MSE across all test samples: {avg_mse:.6f}")

# Save average MSE to summary file
with open(os.path.join(results_dir, "test_summary.txt"), "a") as f:
    f.write(f"\nAverage MSE across all {args.n_samples_test} test samples: {avg_mse:.6f}\n")

# Create a summary of the results
plt.figure(figsize=(12, 8))
plt.text(0.5, 0.95, "Vanilla DeepONet for Chladni Pattern Prediction", 
         horizontalalignment='center', fontsize=18, fontweight='bold')

plt.text(0.5, 0.88, f"Average MSE on test set: {avg_mse:.6f}", 
         horizontalalignment='center', fontsize=16)

plt.text(0.1, 0.78, "Architecture:", fontsize=16, fontweight='bold')
plt.text(0.1, 0.72, f"Branch Network: {branch_net_size}", fontsize=14)
plt.text(0.1, 0.66, f"Trunk Network: {trunk_net_size}", fontsize=14)
plt.text(0.1, 0.60, f"Output Dimension: {args.output_dim}", fontsize=14)

plt.text(0.1, 0.50, "Training:", fontsize=16, fontweight='bold')
plt.text(0.1, 0.44, f"Iterations: {iterations}", fontsize=14)
plt.text(0.1, 0.38, f"Optimizer: Adam (lr={args.learning_rate})", fontsize=14)
plt.text(0.1, 0.32, f"Learning rate decay: inverse time", fontsize=14)
plt.text(0.1, 0.26, f"Batch Size: {args.batch_size if args.batch_size else 'Full batch'}", fontsize=14)

plt.text(0.1, 0.16, "Dataset:", fontsize=16, fontweight='bold')
plt.text(0.1, 0.10, f"Training Samples: {args.n_samples_train}", fontsize=14)
plt.text(0.1, 0.04, f"Testing Samples: {args.n_samples_test}", fontsize=14)

plt.axis('off')
plt.tight_layout()
plt.savefig(f'{args.output_dir}/summary.png', dpi=400, bbox_inches='tight')
plt.close()

print(f"Results saved to {args.output_dir} and {results_dir}")
