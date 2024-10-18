import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import os
import pickle
import torch.nn.functional as F
from models import MLPModel, MLP
from dataclasses import dataclass, field

# ---------------------------
# Dataset Classes
# ---------------------------

class FStarDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # Unpack all seven elements from the dataset
            encoded_map, start, goal, current, g_normalized, h_normalized, target_value = self.data[idx]

            # Normalize positional coordinates by dividing by 127 (assuming map size is 128x128)
            start_normalized = np.array(start) / 127
            goal_normalized = np.array(goal) / 127
            current_normalized = np.array(current) / 127

            # Concatenate all components to form the input tensor
            input_tensor = np.concatenate([
                start_normalized,          # 2 elements
                goal_normalized,           # 2 elements
                current_normalized,        # 2 elements
                [g_normalized, h_normalized],  # 2 elements
                encoded_map                # latent_dim elements
            ])

            # Convert to PyTorch tensors
            input_tensor = torch.from_numpy(input_tensor).float()
            f_star_value_tensor = torch.tensor([target_value]).float()  # Shape: [1]

            # Optional: Add sanity checks
            assert torch.isfinite(input_tensor).all(), "Non-finite values found in input_tensor"
            assert torch.isfinite(f_star_value_tensor), "Non-finite value found in f_star_value_tensor"

            return input_tensor, f_star_value_tensor
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            raise e

# ---------------------------
# Custom Loss Function
# ---------------------------

def custom_loss_function(all_inputs, output, target, lambda1=0.0, lambda2=0.0, lambda3=0.0):
    mse_loss = nn.MSELoss()(output, target)
    
    # Compute gradients of the output with respect to the inputs
    grad_all_inputs = torch.autograd.grad(
        outputs=output,
        inputs=all_inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True
    )[0]
    
    # Assuming that g_normalized and h_normalized are at the end of the input tensor
    grad_g = grad_all_inputs[:, -2]  # g_normalized is second last element
    grad_h = grad_all_inputs[:, -1]  # h_normalized is last element
    
    # Property 1: Ensure gradients are positive
    gradient_loss1 = torch.mean(F.relu(-grad_g)) + torch.mean(F.relu(-grad_h))
    
    # Property 2: Gradient w.r.t h should be greater than gradient w.r.t g
    gradient_loss2 = torch.mean(F.relu(grad_g - grad_h + 1e-6))  # Enforces grad_g ≤ grad_h
    
    # Property 4: Sum of gradients should be at least 2 (Corrected)
    gradient_loss4 = torch.mean(F.relu(2 - grad_g - grad_h))  # Enforces grad_g + grad_h ≥ 2
    
    # Total loss combines MSE loss and penalties for violating properties
    total_loss = mse_loss + lambda1 * gradient_loss1 + lambda2 * gradient_loss2 + lambda3 * gradient_loss4
    return total_loss, mse_loss.item(), gradient_loss1.item(), gradient_loss2.item(), gradient_loss4.item()

# ---------------------------
# Lambda Schedule Function
# ---------------------------

def create_lambda_schedule(lambda_start, lambda_end, epochs, growth_rate):
    def lambda_func(epoch):
        progress = epoch / max(epochs - 1, 1)
        return lambda_start + (lambda_end - lambda_start) * (progress ** growth_rate)
    return lambda_func

# ---------------------------
# Training Functions
# ---------------------------

def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=10, model_path="model.pth",
                loss_fn="mse", criterion=None, lambda_schedules=None, save_each_epoch=False, save_dir="models"):
    print(f"Training model with loss function: {loss_fn}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    if criterion is None:
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        else:
            criterion = custom_loss_function

    best_val_loss = float('inf')
    patience_counter = 0

    # Create directory to save models if it doesn't exist
    if save_each_epoch:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_gradient_loss1 = 0.0
        train_gradient_loss2 = 0.0
        train_gradient_loss4 = 0.0

        # Get current lambdas from the schedules
        if lambda_schedules is not None:
            lambda1 = lambda_schedules['lambda1'](epoch)
            lambda2 = lambda_schedules['lambda2'](epoch)
            lambda3 = lambda_schedules['lambda3'](epoch)
        else:
            lambda1 = lambda2 = lambda3 = 0.0  # Default to zero if no schedule provided

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data.requires_grad_(True)  # Ensure gradients can be computed
            optimizer.zero_grad()
            output, all_inputs = model(data)

            if loss_fn == "mse":
                loss = criterion(output, target)
                mse_loss_val = loss.item()
                gradient_loss1_val = gradient_loss2_val = gradient_loss4_val = 0.0
            else:
                loss, mse_loss_val, gradient_loss1_val, gradient_loss2_val, gradient_loss4_val = criterion(
                    all_inputs, output, target, lambda1, lambda2, lambda3
                )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mse_loss += mse_loss_val
            train_gradient_loss1 += gradient_loss1_val
            train_gradient_loss2 += gradient_loss2_val
            train_gradient_loss4 += gradient_loss4_val

        train_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        train_gradient_loss1 /= len(train_loader)
        train_gradient_loss2 /= len(train_loader)
        train_gradient_loss4 /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                loss = nn.MSELoss()(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, '
              f'MSE Loss: {train_mse_loss:.6f}, '
              f'Grad1 Loss: {train_gradient_loss1:.6f}, '
              f'Grad2 Loss: {train_gradient_loss2:.6f}, '
              f'Grad4 Loss: {train_gradient_loss4:.6f}, '
              f'Val Loss: {val_loss:.6f}, LR: {current_lr}')
        if lambda_schedules is not None:
            print(f'    Current Lambdas: lambda1={lambda1:.4f}, lambda2={lambda2:.4f}, lambda3={lambda3:.4f}')

        scheduler.step(val_loss)

        # Save the model at each epoch if flag is set
        if save_each_epoch:
            epoch_model_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            print(f"Model saved at {epoch_model_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved to {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print(f"Best validation loss: {best_val_loss:.6f}")
    return model


# ---------------------------
# Argument Parser
# ---------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train F* Prediction Model using a Generated Dataset")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the generated dataset pickle file")
    parser.add_argument("--norm_path", type=str, required=True,
                        help="Path to the normalization values pickle file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--latent_dim", type=int, default=512,
                        help="Dimension for autoencoder vector")
    parser.add_argument("--loss_function", type=str, choices=[
                        'mse', 'custom'], default='mse', help="Choose between MSE and custom loss function")
    parser.add_argument("--model_save_path", type=str,
                        default="model.pth", help="Path to save the best model")
    parser.add_argument("--learning_type", type=str, choices=[
                        'heuristic', 'priority'], default='priority', help="Choose the type of function that is being learned.")
    parser.add_argument("--lambda1_start", type=float, default=0.05, help="Starting value of lambda1")
    parser.add_argument("--lambda1_end", type=float, default=.8, help="Ending value of lambda1")
    parser.add_argument("--lambda2_start", type=float, default=0.05, help="Starting value of lambda2")
    parser.add_argument("--lambda2_end", type=float, default=.8, help="Ending value of lambda2")
    parser.add_argument("--lambda3_start", type=float, default=0.05, help="Starting value of lambda3")
    parser.add_argument("--lambda3_end", type=float, default=.8, help="Ending value of lambda3")
    parser.add_argument("--growth_rate", type=float, default=2.0, help="Growth rate for exponential increase of lambdas")
    parser.add_argument("--save_each_epoch", action='store_true', help="Save model at each epoch")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models at each epoch")
    return parser.parse_args()

# ---------------------------
# Main Function
# ---------------------------

def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load normalization values
    if not os.path.isfile(args.norm_path):
        raise FileNotFoundError(f"Normalization values not found at {args.norm_path}")
    
    with open(args.norm_path, 'rb') as f:
        normalization_values = pickle.load(f)
    print(f"Loaded normalization values from {args.norm_path}")

    # Load dataset
    if not os.path.isfile(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")
    
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded dataset from {args.dataset_path}")

    full_dataset = FStarDataset(dataset)

    # Split into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, pin_memory=True, num_workers=4)

    # Initialize MLP Model
    output_size = 1
    input_size = (3 * 2) + args.latent_dim + 2  # start, goal, current, g_normalized, h_normalized
    print(f"Calculated input size for MLP: {input_size}")
    model = MLPModel(input_size, output_size).to(device)

    # Choose training type based on arguments
    if args.learning_type in ['heuristic', 'priority']:
        print(f"Training MLPModel with learning type: {args.learning_type}")
    else:
        print(f"Invalid learning type: {args.learning_type}. Choose 'heuristic' or 'priority'.")
        return

    # Create lambda schedules if using custom loss
    if args.loss_function == "custom":
        lambda_schedules = {
            'lambda1': create_lambda_schedule(args.lambda1_start, args.lambda1_end, args.epochs, args.growth_rate),
            'lambda2': create_lambda_schedule(args.lambda2_start, args.lambda2_end, args.epochs, args.growth_rate),
            'lambda3': create_lambda_schedule(args.lambda3_start, args.lambda3_end, args.epochs, args.growth_rate)
        }
    else:
        lambda_schedules = None

    # Train the MLP Model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        model_path=args.model_save_path,
        loss_fn=args.loss_function,
        criterion=None,  # Let train_model handle setting the criterion
        lambda_schedules=lambda_schedules,
        save_each_epoch=args.save_each_epoch,
        save_dir=args.save_dir
    )

    print(f"Training completed. Best model saved as {args.model_save_path}")

if __name__ == '__main__':
    main()
