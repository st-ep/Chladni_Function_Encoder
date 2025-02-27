import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Define the Basis-to-Basis mapping neural network
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
parser.add_argument("--dataset_path", type=str, required=True, 
                    help="Path to the saved representation dataset")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--test_size", type=float, default=0.1, 
                    help="Proportion of data to use for validation")
parser.add_argument("--hidden_dims", type=str, default="256,512,256", 
                    help="Comma-separated list of hidden layer dimensions")
parser.add_argument("--output_dir", type=str, default="b2b_model")
parser.add_argument("--seed", type=int, default=42)
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

# Load the dataset
print(f"Loading dataset from {args.dataset_path}")
data = torch.load(args.dataset_path)
s_representations = data['s_representations']
z_representations = data['z_representations']

# Remove the middle dimension if it's 1
if s_representations.shape[1] == 1:
    s_representations = s_representations.squeeze(1)
if z_representations.shape[1] == 1:
    z_representations = z_representations.squeeze(1)

print(f"S representations shape: {s_representations.shape}")
print(f"Z representations shape: {z_representations.shape}")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    s_representations, z_representations, 
    test_size=args.test_size, 
    random_state=args.seed
)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    shuffle=False
)

# Parse hidden dimensions
hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]

# Create the model
input_dim = s_representations.shape[1]
output_dim = z_representations.shape[1]
model = BasisToBasicsNetwork(input_dim, output_dim, hidden_dims).to(device)
print(f"Created model with input dim {input_dim}, output dim {output_dim}, and hidden dims {hidden_dims}")

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training loop
best_val_loss = float('inf')
train_losses = []
val_losses = []

print(f"Starting training for {args.epochs} epochs...")
for epoch in range(args.epochs):
    # Training
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Train)", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Val)", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            val_loss += loss.item() * X_batch.size(0)
    
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    
    # Print progress
    print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dims': hidden_dims,
        }, f"{args.output_dir}/best_model.pt")
        print(f"Saved new best model with validation loss: {val_loss:.6f}")

# Plot training curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{args.output_dir}/loss_curve.png", dpi=300)

# Evaluate on test set
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        
        # Forward pass
        y_pred = model(X_batch).cpu()
        
        all_predictions.append(y_pred)
        all_targets.append(y_batch)

all_predictions = torch.cat(all_predictions, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Compute average coefficient-wise error
coef_errors = torch.mean((all_predictions - all_targets)**2, dim=0)
avg_error = torch.mean(coef_errors)
print(f"Average test set MSE: {avg_error:.6f}")

# Visualize predictions for a few samples
plt.figure(figsize=(15, 12))
n_samples = min(5, len(all_predictions))

for i in range(n_samples):
    plt.subplot(n_samples, 2, i*2 + 1)
    plt.plot(all_targets[i].numpy(), label='True')
    plt.plot(all_predictions[i].numpy(), label='Predicted')
    plt.title(f'Sample {i+1} - Z Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot residuals
    plt.subplot(n_samples, 2, i*2 + 2)
    plt.plot(all_predictions[i].numpy() - all_targets[i].numpy())
    plt.title(f'Sample {i+1} - Residuals')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Predicted - True')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True)

plt.tight_layout()
plt.savefig(f"{args.output_dir}/prediction_examples.png", dpi=300)

# Visualize coefficient-wise errors
plt.figure(figsize=(10, 6))
plt.bar(range(len(coef_errors)), coef_errors.numpy())
plt.title('Coefficient-wise Mean Squared Error')
plt.xlabel('Coefficient Index')
plt.ylabel('MSE')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(f"{args.output_dir}/coefficient_errors.png", dpi=300)

print(f"Training complete. Results saved to {args.output_dir}")
