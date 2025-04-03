import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import h5py
from tqdm import tqdm

from ChladniDataset import ChladniDataset  # For S_train
from ChladniDataset_u import ChladniDataset as ChladniDataset_Z  # For Z_train

# Fourier Layer implementation
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, x direction
        self.modes2 = modes2  # Number of Fourier modes to multiply, y direction

        # Create weights for the Fourier layer
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def compl_mul2d(self, input, weights):
        # Complex multiplication 
        # (batch, in_channel, x, y) * (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Prepare the output tensor of the right size
        out_ft = torch.zeros(batchsize, self.out_channels, 
                             x.size(-2), x.size(-1)//2 + 1, 
                             dtype=torch.cfloat, device=x.device)
        
        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], 
                            torch.view_as_complex(self.weights1))
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# FNO architecture
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels=1, out_channels=1, n_layers=4):
        super(FNO2d, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # Initial projection layer
        self.fc0 = nn.Linear(in_channels, self.width)
        
        # Fourier layers
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for i in range(self.n_layers):
            self.conv_layers.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.w_layers.append(nn.Conv2d(self.width, self.width, 1))
            
        # Final output layer
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # Assume x is of shape (batch, in_channels, nx, ny)
        # First channel dimension needed for FNO
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        nx = x.shape[2]
        ny = x.shape[3]
        
        # Initial projection layer
        x_linear = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Fourier layers
        x = x_linear
        for i in range(self.n_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            x = F.gelu(x)
        
        # Final output layer
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--modes1", type=int, default=12, 
                    help="Number of Fourier modes (x direction)")
parser.add_argument("--modes2", type=int, default=12, 
                    help="Number of Fourier modes (y direction)")
parser.add_argument("--width", type=int, default=32, 
                    help="Width of the FNO network")
parser.add_argument("--n_layers", type=int, default=4, 
                    help="Number of Fourier layers")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--scheduler_step", type=int, default=100)
parser.add_argument("--scheduler_gamma", type=float, default=0.5)
parser.add_argument("--n_samples_train", type=int, default=-1, 
                    help="Number of samples for training, use -1 for all available samples")
parser.add_argument("--n_samples_test", type=int, default=200,
                    help="Number of samples for testing")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="fno_results")
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

print(f"""
Hyperparameters:
- Epochs: {args.epochs}
- Learning Rate: {args.learning_rate}
- FNO Modes: {args.modes1} x {args.modes2}
- FNO Width: {args.width}
- FNO Layers: {args.n_layers}
- Batch Size: {args.batch_size}
- Training Samples: {args.n_samples_train}
- Testing Samples: {args.n_samples_test}
""")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load datasets
s_dataset = ChladniDataset(n_functions=args.n_samples_train + args.n_samples_test)
z_dataset = ChladniDataset_Z(n_functions=args.n_samples_train + args.n_samples_test)

print(f"Dataset contains {s_dataset.n_functions} samples")
print(f"S shape: {s_dataset.S_train.shape}")
print(f"Z shape: {z_dataset.Z_train.shape}")
print(f"Grid shape: {s_dataset.grid.shape}")

# Prepare data for FNO
def prepare_data_for_fno():
    """Prepare data in the format required by FNO"""
    # Get the grid and force/displacement data
    grid = s_dataset.grid
    
    # Determine grid size (assuming square grid)
    n_side = int(np.sqrt(s_dataset.num_points))
    
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
        s_data = s_dataset.S_train
        z_data = z_dataset.Z_train
        
        # Use all available training samples if n_samples_train is -1
        n_train = len(s_data) if args.n_samples_train == -1 else min(args.n_samples_train, len(s_data))
        n_test = min(args.n_samples_test, S_test.shape[0])
        
        print(f"Using {n_train} training samples and {n_test} test samples")
        
        # Reshape data for FNO (batch, channels, height, width)
        x_train_data = []
        y_train_data = []
        
        for i in range(n_train):
            # Reshape to 2D grid format
            x = s_data[i].reshape(1, n_side, n_side)
            y = z_data[i].reshape(1, n_side, n_side)
            
            x_train_data.append(x)
            y_train_data.append(y)
        
        x_test_data = []
        y_test_data = []
        
        for i in range(n_test):
            # Reshape to 2D grid format
            x = S_test[i].reshape(1, n_side, n_side)
            y = Z_test[i].reshape(1, n_side, n_side)
            
            x_test_data.append(x)
            y_test_data.append(y)
        
        x_train = torch.stack(x_train_data).to(device)
        y_train = torch.stack(y_train_data).to(device)
        x_test = torch.stack(x_test_data).to(device)
        y_test = torch.stack(y_test_data).to(device)
        
        # Combine for later visualization
        s_data_all = torch.cat([s_data[:n_train], S_test[:n_test]], dim=0)
        
    else:
        # Original code path - split training data into train/test
        print("Splitting training data into train/test sets...")
        s_data = s_dataset.S_train
        z_data = z_dataset.Z_train
        
        # Use all available samples for training minus test samples if n_samples_train is -1
        total_available = len(s_data)
        if args.n_samples_train == -1:
            n_train = max(total_available - args.n_samples_test, 0)
        else:
            n_train = min(args.n_samples_train, total_available - args.n_samples_test)
        n_test = min(args.n_samples_test, total_available - n_train)
        
        print(f"Using {n_train} training samples and {n_test} test samples out of {total_available} total samples")
        
        # Reshape data for FNO (batch, channels, height, width)
        x_data = []
        y_data = []
        
        for i in range(n_train + n_test):
            # Reshape to 2D grid format
            x = s_data[i].reshape(1, n_side, n_side)
            y = z_data[i].reshape(1, n_side, n_side)
            
            x_data.append(x)
            y_data.append(y)
        
        x_data = torch.stack(x_data)
        y_data = torch.stack(y_data)
        
        # Split into train and test
        x_train = x_data[:n_train].to(device)
        y_train = y_data[:n_train].to(device)
        x_test = x_data[n_train:n_train+n_test].to(device)
        y_test = y_data[n_train:n_train+n_test].to(device)
        
        # For visualization
        s_data_all = s_data[:n_train+n_test]
    
    print(f"Training data shape: {x_train.shape} -> {y_train.shape}")
    print(f"Testing data shape: {x_test.shape} -> {y_test.shape}")
    
    return x_train, y_train, x_test, y_test, s_data_all

# Prepare data
x_train, y_train, x_test, y_test, s_data_all = prepare_data_for_fno()

# Define loss function
criterion = nn.MSELoss()

# Create FNO model
model = FNO2d(modes1=args.modes1, 
              modes2=args.modes2, 
              width=args.width,
              in_channels=1, 
              out_channels=1, 
              n_layers=args.n_layers).to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                           step_size=args.scheduler_step, 
                                           gamma=args.scheduler_gamma)

# Create a directory for model checkpoints
checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')

# Progress tracking function
def print_progress(epoch, train_loss, val_loss, best_epoch, best_val_loss, start_time):
    elapsed = datetime.now() - start_time
    progress = min(100, epoch * 100 // args.epochs)
    bar_length = 30
    filled_length = int(bar_length * progress // 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f"\r[{bar}] {progress}% | Epoch: {epoch}/{args.epochs} | "
          f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
          f"Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch}) | "
          f"Time: {elapsed}", end="")

# Training loop
print("\nStarting training...")
best_epoch = 0
start_time = datetime.now()

for epoch in range(1, args.epochs + 1):
    model.train()
    
    # Create data loader
    indices = torch.randperm(x_train.size(0))
    
    # Adjust batch size for final batch
    total_loss = 0
    batches = 0
    
    for i in range(0, x_train.size(0), args.batch_size):
        batch_indices = indices[i:min(i + args.batch_size, x_train.size(0))]
        
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(x_batch)
        
        # Calculate loss
        loss = criterion(y_pred, y_batch)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
    
    # Average training loss
    train_loss = total_loss / batches
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        val_loss = criterion(y_pred, y_test).item()
    
    val_losses.append(val_loss)
    
    # Update learning rate
    scheduler.step()
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, os.path.join(checkpoint_dir, 'best_model.pt'))
        
    
    # Print progress every 10 epochs or at start/end
    if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
        print_progress(epoch, train_loss, val_loss, best_epoch, best_val_loss, start_time)
    
    # Save checkpoint every 100 epochs
    if epoch % 100 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt'))

print("\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
print(f"Total time: {datetime.now() - start_time}")

# Load the best model
checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']}")

# Plot loss curves
plt.figure(figsize=(12, 8))
plt.semilogy(range(1, args.epochs + 1), train_losses, label='Train Loss')
plt.semilogy(range(1, args.epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.grid(True, which="both", ls="--")
plt.legend(fontsize=14)
plt.title('Training and Validation Loss', fontsize=18)
plt.savefig(os.path.join(args.output_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create directories for test/train results
test_results_dir = os.path.join(args.output_dir, "test_results")
os.makedirs(test_results_dir, exist_ok=True)

train_results_dir = os.path.join(args.output_dir, "train_results")
os.makedirs(train_results_dir, exist_ok=True)

# Visualize results on test data
model.eval()
n_side = int(np.sqrt(s_dataset.num_points))

# Create a summary file for test results
with open(os.path.join(test_results_dir, "test_summary.txt"), "w") as f:
    f.write(f"FNO Test Results\n")
    f.write(f"===============\n\n")
    f.write(f"Model Architecture:\n")
    f.write(f"  Modes: {args.modes1} x {args.modes2}\n")
    f.write(f"  Width: {args.width}\n")
    f.write(f"  Layers: {args.n_layers}\n\n")

# Create a summary file for training results
with open(os.path.join(train_results_dir, "train_summary.txt"), "w") as f:
    f.write(f"FNO Training Results\n")
    f.write(f"==================\n\n")
    f.write(f"Model Architecture:\n")
    f.write(f"  Modes: {args.modes1} x {args.modes2}\n")
    f.write(f"  Width: {args.width}\n")
    f.write(f"  Layers: {args.n_layers}\n\n")

# Visualize a few training samples
n_train_display = min(4, args.n_samples_train)
print(f"Visualizing {n_train_display} training samples...")

for i in range(n_train_display):
    print(f"Visualizing training sample {i+1}/{n_train_display}")
    
    # Get training data
    x_input = x_train[i:i+1]
    s_input_img = x_input[0, 0].cpu().numpy()
    
    # Predict using FNO
    with torch.no_grad():
        y_pred = model(x_input)
    
    # Get ground truth
    y_true = y_train[i]
    
    # Convert to numpy for plotting
    y_pred_img = y_pred[0, 0].cpu().numpy()
    y_true_img = y_true[0].cpu().numpy()
    
    # Calculate error
    mse = np.mean((y_pred_img - y_true_img)**2)
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
    axs[0].set_title('Predicted Z (FNO)', fontsize=16)
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
        axs[1].set_title('Predicted Chladni Pattern (FNO)', color='white', fontsize=16)
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

# Evaluate on test data
n_test_display = min(4, args.n_samples_test)
total_test_mse = 0

for i in range(n_test_display):
    print(f"Evaluating test sample {i+1}/{n_test_display}")
    
    # Get test data
    x_input = x_test[i:i+1]
    s_input_img = x_input[0, 0].cpu().numpy()
    
    # Predict using FNO
    with torch.no_grad():
        y_pred = model(x_input)
    
    # Get ground truth
    y_true = y_test[i]
    
    # Convert to numpy for plotting
    y_pred_img = y_pred[0, 0].cpu().numpy()
    y_true_img = y_true[0].cpu().numpy()
    
    # Calculate error
    mse = np.mean((y_pred_img - y_true_img)**2)
    total_test_mse += mse
    print(f"Test sample {i+1} MSE: {mse:.6f}")
    
    # Save result to summary file
    with open(os.path.join(test_results_dir, "test_summary.txt"), "a") as f:
        f.write(f"Sample {i+1} MSE: {mse:.6f}\n")
    
    # 0. Input force field
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.contourf(s_input_img, levels=50, cmap='viridis')
    ax.set_title('Input Force Field (S)', fontsize=16)
    ax.set_xlabel('X coordinate', fontsize=14)
    ax.set_ylabel('Y coordinate', fontsize=14)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(f'{test_results_dir}/input_force_{i + 1}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    # 1. Displacement field comparison
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Use same color scale for both plots
    vmin = min(y_pred_img.min(), y_true_img.min())
    vmax = max(y_pred_img.max(), y_true_img.max())
    
    im = axs[0].contourf(y_pred_img, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('Predicted Z (FNO)', fontsize=16)
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
    fig.savefig(f'{test_results_dir}/displacement_{i + 1}.png', dpi=400, bbox_inches='tight')
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
        axs[1].set_title('Predicted Chladni Pattern (FNO)', color='white', fontsize=16)
        axs[1].set_facecolor('black')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
    except Exception as e:
        print(f"Warning: Could not plot predicted Chladni pattern contour: {str(e)}")
        axs[1].text(0.5, 0.5, 'No zero contour found', 
                   horizontalalignment='center', 
                   color='white', fontsize=14)
    
    plt.tight_layout()
    fig.savefig(f'{test_results_dir}/chladni_pattern_{i + 1}.png', dpi=400, bbox_inches='tight')
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
    fig.savefig(f'{test_results_dir}/cross_section_{i + 1}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

# Calculate average test error on all test samples
with torch.no_grad():
    total_loss = 0
    for i in range(0, x_test.size(0), args.batch_size):
        x_batch = x_test[i:min(i + args.batch_size, x_test.size(0))]
        y_batch = y_test[i:min(i + args.batch_size, y_test.size(0))]
        y_pred = model(x_batch)
        batch_loss = criterion(y_pred, y_batch).item()
        total_loss += batch_loss * x_batch.size(0)
    
    avg_test_mse = total_loss / x_test.size(0)

print(f"\nAverage MSE across all test samples: {avg_test_mse:.6f}")

# Save average MSE to summary file
with open(os.path.join(test_results_dir, "test_summary.txt"), "a") as f:
    f.write(f"\nAverage MSE across all {args.n_samples_test} test samples: {avg_test_mse:.6f}\n")

# Create a summary of the results
plt.figure(figsize=(12, 8))
plt.text(0.5, 0.95, "Fourier Neural Operator for Chladni Pattern Prediction", 
         horizontalalignment='center', fontsize=18, fontweight='bold')

plt.text(0.5, 0.88, f"Average MSE on test set: {avg_test_mse:.6f}", 
         horizontalalignment='center', fontsize=16)

plt.text(0.1, 0.78, "Architecture:", fontsize=16, fontweight='bold')
plt.text(0.1, 0.72, f"FNO Modes: {args.modes1} x {args.modes2}", fontsize=14)
plt.text(0.1, 0.66, f"FNO Width: {args.width}", fontsize=14)
plt.text(0.1, 0.60, f"FNO Layers: {args.n_layers}", fontsize=14)

plt.text(0.1, 0.50, "Training:", fontsize=16, fontweight='bold')
plt.text(0.1, 0.44, f"Epochs: {args.epochs}", fontsize=14)
plt.text(0.1, 0.38, f"Optimizer: Adam (lr={args.learning_rate})", fontsize=14)
plt.text(0.1, 0.32, f"Scheduler: Step (step={args.scheduler_step}, gamma={args.scheduler_gamma})", fontsize=14)
plt.text(0.1, 0.26, f"Batch Size: {args.batch_size}", fontsize=14)

plt.text(0.1, 0.16, "Dataset:", fontsize=16, fontweight='bold')
plt.text(0.1, 0.10, f"Training Samples: {args.n_samples_train}", fontsize=14)
plt.text(0.1, 0.04, f"Testing Samples: {args.n_samples_test}", fontsize=14)

plt.axis('off')
plt.tight_layout()
plt.savefig(f'{args.output_dir}/summary.png', dpi=400, bbox_inches='tight')
plt.close()

print(f"Results saved to {args.output_dir}")

# If we want to run the model with parameters from command line:
if __name__ == "__main__":
    print("FNO.py completed successfully")
