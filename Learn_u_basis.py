import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import gc

from FunctionEncoder import FunctionEncoder
from FunctionEncoder import MSECallback, ListCallback, TensorboardCallback
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from ChladniDataset_u import ChladniDataset   # use our updated dataset for Z values

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=110)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=250000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()

epochs = args.epochs
n_basis = args.n_basis
lr = args.lr
train_method = args.train_method
load_path = args.load_path
residuals = args.residuals
arch = 'MLP'

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"""
Hyperparameters:
- Epochs: {epochs}
- Number of Basis Functions: {n_basis}
- Learning Rate: {lr}
- Training Method: {train_method}
- Load Path: {load_path}
- Residuals: {residuals}
- Architecture: {arch}
""")

device = "cuda" if torch.cuda.is_available() else "cpu"
print('device: ', device)

if load_path is None:
    logdir = f"parameterized_Chladni_Z_EncoderOnly/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
    print("loading a trained model...")

# Before training or large operations
torch.cuda.empty_cache()
gc.collect()

# Reduce batch size or add memory management
dataset = ChladniDataset(n_functions=100)  
dataset_eval = ChladniDataset(n_functions=100)  # Reduce from 512 to smaller batch

# Optional: Set memory allocation strategy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

if load_path is None:
    print("training a new model...")
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))
    
    # update learning rate
    for param_group in model.opt.param_groups:
        param_group['lr'] = lr
    print('lr: ', model.opt.param_groups[0]['lr'])
    
    # create callbacks
    class LossTracker(BaseCallback):
        def __init__(self):
            super().__init__()
            self.losses = []
            
        def on_training_start(self, locals: dict) -> None:
            pass
            
        def on_training_end(self, locals: dict) -> None:
            pass
            
        def on_step(self, locals: dict) -> None:
            if "prediction_loss" in locals:
                self.losses.append(locals["prediction_loss"].item())

    cb1 = TensorboardCallback(logdir)
    cb2 = MSECallback(dataset, tensorboard=cb1.tensorboard)
    cb3 = LossTracker()
    callback = ListCallback([cb1, cb2, cb3])
    
    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)
    
    # Wait for tensorboard to finish writing
    cb1.tensorboard.flush()
    
    # Create a figure for the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(cb3.losses, label='Training Loss')  # Use our LossTracker's losses
    plt.xlabel('Epoch')
    plt.ylabel('Mean Distance Squared')
    plt.yscale('log')  # Use log scale since loss values can span multiple orders of magnitude
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{logdir}/loss_history.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
    print('saving the model...')
else:
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# For visualization, we pick one training sample and use the full grid from the dataset.
with torch.no_grad():
    # Create 4 separate plots for different samples
    for sample_idx in range(4):
        # Get the full grid of coordinate pairs
        grid = dataset.grid  
        total_points = dataset.num_points
        
        # For computing the representation, select a support set from the current sample
        n_examples = dataset.n_examples
        indices = torch.randperm(total_points)[:n_examples]
        support_xs = grid[indices].unsqueeze(0)  # shape: (1, n_examples, 2)
        
        # Get corresponding Z_train values from the current sample
        # Use reshape instead of view to handle non-contiguous tensors
        Z_train_flat = dataset.Z_train[sample_idx].reshape(-1, 1)  # shape: (total_points, 1)
        support_Z = Z_train_flat[indices].unsqueeze(0)   # shape: (1, n_examples, 1)
        
        representations, _ = model.compute_representation(support_xs, support_Z)
        
        # Predict on the full grid
        grid_batch = grid.unsqueeze(0)  # shape: (1, total_points, 2)
        Z_pred = model.predict(grid_batch, representations)  # shape: (1, total_points, 1)
        
        # Reshape predictions and ground truth to square images for contour plotting
        num_side = int(np.sqrt(total_points))
        Z_pred_img = Z_pred.reshape(num_side, num_side).detach().cpu().numpy()
        Z_true_img = dataset.Z_train[sample_idx].reshape(num_side, num_side, 1).squeeze(-1).detach().cpu().numpy()
        
        # Create a new figure for each sample
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        im = axs[0].contourf(Z_pred_img, levels=50, cmap='viridis')
        axs[0].set_title('Predicted Z (Displacement)')
        fig.colorbar(im, ax=axs[0])
        
        im = axs[1].contourf(Z_true_img, levels=50, cmap='viridis')
        axs[1].set_title('True Z (Displacement)')
        fig.colorbar(im, ax=axs[1])
        
        im = axs[2].contourf(np.abs(Z_true_img - Z_pred_img), levels=50, cmap='viridis')
        axs[2].set_title('Absolute Error')
        fig.colorbar(im, ax=axs[2])
        
        plt.suptitle(f'Sample {sample_idx + 1}')
        fig.savefig(f'{logdir}/results{sample_idx + 1}.png', dpi=400, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        
        # Also plot the zero contour lines (Chladni patterns)
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
        fig.savefig(f'{logdir}/chladni_pattern{sample_idx + 1}.png', dpi=400, bbox_inches='tight')
        plt.close(fig)

# Final evaluation error on a larger batch
with torch.no_grad():
    example_xs, example_Z, query_xs, query_Z, _ = dataset_eval.sample()
    example_xs = example_xs.to(device)
    example_Z = example_Z.to(device)
    representations, _ = model.compute_representation(example_xs, example_Z)
    Z_preds = model.predict(query_xs, representations)  # shape: (n_functions, n_points, 1)
    error = torch.mean((Z_preds - query_Z)**2)
    print('final error: ', error.item())