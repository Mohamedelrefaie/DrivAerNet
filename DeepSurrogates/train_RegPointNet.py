#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu, mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

"""
import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from DeepSurrogate_models import RegPointNet
import pandas as pd
from DrivAerNetDataset import DrivAerNetDataset
# Configuration dictionary to hold hyperparameters and settings
config = {
    'exp_name': 'DragPrediction_DrivAerNet_PointNet_r2_batchsize32_200epochs_100kpoints_tsne_NeurIPS',
    'cuda': True,
    'seed': 1,
    'num_points': 100000,
    'lr': 0.001,
    'batch_size':32,
    'epochs': 30,
    'dropout': 0.0,
    'emb_dims': 1024,
    'k': 40,
    'optimizer': 'adam',
    'channels': [6, 64, 128, 256, 512, 1024],
    'linear_sizes': [128, 64, 32, 16],
    'output_channels': 1,
    'dataset_path': '../DrivAerNet_FEN_Processed_Point_Clouds_100k',  # Update this with your dataset path
    'aero_coeff': '../DrivAerNetPlusPlus_Cd_8k_Updated.csv',
    'subset_dir': '../subset_dir'

}

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")

def setup_seed(seed: int):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def r2_score(output, target):
    """Compute R-squared score."""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def initialize_model(config: dict) -> torch.nn.Module:
    """
    Initialize and return the RegDGCNN model.
    Args:
        config (dict): A dictionary containing configuration parameters for the model, including:
            - k: The number of nearest neighbors to consider in the graph construction.
            - channels: A list defining the number of output channels for each graph convolutional layer.
            - linear_sizes: A list defining the sizes of each linear layer following the convolutional layers.
            - emb_dims: The size of the global feature vector obtained after the graph convolutional and pooling layers.
            - dropout: The dropout rate applied after each linear layer for regularization.
            - output_channels: The number of output channels in the final layer, equal to the number of prediction targets.

    Returns:
        torch.nn.Module: The initialized RegDGCNN model, potentially wrapped in a DataParallel module if multiple GPUs are used.
    """

    # Instantiate the RegPointNet model with the specified configuration parameters
    model = RegPointNet(args=config).to(device)
    #model = PointTransformer().to(device)
    # If CUDA is enabled and more than one GPU is available, wrap the model in a DataParallel module
    # to enable parallel computation across multiple GPUs. Specifically, use GPUs with IDs 0, 1, 2, and 3.
    if config['cuda'] and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 2, 3])

    # Return the initialized model
    return model


def get_dataloaders(dataset_path: str, aero_coeff: str, subset_dir: str, num_points: int, batch_size: int,
                    train_frac: float = 1.0) -> tuple:
    """
    Prepare and return the training, validation, and test DataLoader objects.

    Args:
        dataset_path (str): The file path to the dataset directory containing the STL files.
        aero_coeff (str): The path to the CSV file with metadata for the models.
        subset_dir (str): The directory containing the subset files (train, val, test).
        num_points (int): The number of points to sample from each point cloud in the dataset.
        batch_size (int): The number of samples per batch to load.
        train_frac (float): Fraction of the training data to be used for training.

    Returns:
        tuple: A tuple containing the training DataLoader, validation DataLoader, and test DataLoader.
    """
    # Initialize the full dataset
    full_dataset = DrivAerNetDataset(root_dir=dataset_path, csv_file=aero_coeff, num_points=num_points, pointcloud_exist=True)

    # Helper function to create subsets from IDs in text files
    def create_subset(dataset, ids_file):
        try:
            with open(os.path.join(subset_dir, ids_file), 'r') as file:
                subset_ids = file.read().split()
            subset_indices = dataset.data_frame[dataset.data_frame['Design'].isin(subset_ids)].index.tolist()
            return Subset(dataset, subset_indices)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")

    # Create training subset using the corresponding subset file
    train_dataset = create_subset(full_dataset, 'train_design_ids.txt')

    # Reduce the size of the training dataset if train_frac is less than 1.0
    if train_frac < 1.0:
        train_size = int(len(train_dataset) * train_frac)
        train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])

    # Initialize DataLoaders for each subset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_dataset = create_subset(full_dataset, 'val_design_ids.txt')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_dataset = create_subset(full_dataset, 'test_design_ids.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader


def train_and_evaluate(model: torch.nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, config: dict):
    """
    Train and evaluate the model using the provided dataloaders and configuration. This function handles the training
    loop, including forward and backward propagation, and evaluates the model's performance on the validation set at
    the end of each epoch. It saves the best model based on the lowest validation loss and also saves the final model
    state at the end of all epochs.

    Args:
        model (torch.nn.Module): The model to be trained and evaluated.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        config (dict): Configuration dictionary containing settings such as learning rate, batch size, number of epochs,
                       optimizer choice, etc.

    Returns:
        tuple: A tuple containing paths to the best and final saved model states.
    """
    # Initialize lists to store loss values for plotting or analysis
    train_losses, val_losses = [], []

    # Record the start time of training for performance analysis
    training_start_time = time.time()

    # Initialize the optimizer based on configuration; default to Adam if 'adam' is specified, else use SGD
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)

    # Initialize a learning rate scheduler to adjust the learning rate based on validation loss performance
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.1, verbose=True)

    # Best model tracking variables
    best_mse = float('inf')
    best_model_path = None

    # Training loop
    for epoch in range(config['epochs']):
        # Timing each epoch for performance analysis
        epoch_start_time = time.time()
        model.train()  # Set model to training mode
        total_loss, total_r2 = 0, 0

        # Iterate over batches of data
        for data, targets in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Training]"):
            data, targets = data.to(device), targets.to(device).squeeze()
            data = data.permute(0, 2, 1)  # Adjust data dimensions if necessary

            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = F.mse_loss(outputs.squeeze(), targets)
            r2 = r2_score(outputs.squeeze(), targets)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Aggregate statistics
            total_loss += loss.item()
            total_r2 += r2.item()

        # Calculate average loss and R² for the epoch
        avg_loss = total_loss / len(train_dataloader)
        avg_r2 = total_r2 / len(train_dataloader)
        train_losses.append(avg_loss)

        # Epoch summary
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.6f} Time: {epoch_duration:.2f}s")
        print(f"Average Training R²: {avg_r2:.4f}")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss, val_r2 = 0, 0
        inference_times = []

        with torch.no_grad():
            for data, targets in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Validation]"):
                inference_start_time = time.time()
                data, targets = data.to(device), targets.to(device).squeeze()
                data = data.permute(0, 2, 1)

                outputs = model(data)
                loss = F.mse_loss(outputs.squeeze(), targets)
                val_loss += loss.item()
                val_r2 += r2_score(outputs.squeeze(), targets).item()
                inference_duration = time.time() - inference_start_time
                inference_times.append(inference_duration)

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        avg_inference_time = sum(inference_times) / len(inference_times)

        # Validation summary
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}, Avg Inference Time: {avg_inference_time:.4f}s")
        print(f"Average Validation R²: {val_r2 / len(val_dataloader):.4f}")

        # Update the best model if the current model outperforms previous models
        if avg_val_loss < best_mse:
            best_mse = avg_val_loss
            best_model_path = os.path.join('models', f'{config["exp_name"]}_best_model.pth')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with MSE: {best_mse:.6f}")

        scheduler.step(avg_val_loss)  # Update the learning rate based on the validation loss

    # Final model saving
    final_model_path = os.path.join('models', f'{config["exp_name"]}_final_model.pth')
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")

    # Save training and validation loss histories
    np.save(os.path.join('models', f'{config["exp_name"]}_train_losses.npy'), np.array(train_losses))
    np.save(os.path.join('models', f'{config["exp_name"]}_val_losses.npy'), np.array(val_losses))

    return best_model_path, final_model_path


def test_model(model: torch.nn.Module, test_dataloader: DataLoader, config: dict):
    """
    Test the model using the provided test DataLoader and calculate different metrics.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        test_dataloader (DataLoader): DataLoader for the test set.
        config (dict): Configuration dictionary containing model settings.

    """
    model.eval()  # Set the model to evaluation mode
    total_mse, total_mae, total_r2 = 0, 0, 0
    max_mae = 0
    total_inference_time = 0  # To track total inference time
    total_samples = 0  # To count the total number of samples processed

    # Disable gradient calculation
    with torch.no_grad():
        for data, targets in test_dataloader:
            start_time = time.time()  # Start time for inference

            data, targets = data.to(device), targets.to(device).squeeze()
            data = data.permute(0, 2, 1)
            outputs = model(data)

            end_time = time.time()  # End time for inference
            inference_time = end_time - start_time
            total_inference_time += inference_time  # Accumulate total inference time

            mse = F.mse_loss(outputs.squeeze(), targets) #Mean Squared Error (MSE)
            mae = F.l1_loss(outputs.squeeze(), targets) #Mean Absolute Error (MAE),
            r2 = r2_score(outputs.squeeze(), targets) #R-squared

            # Accumulate metrics to compute averages later
            total_mse += mse.item()
            total_mae += mae.item()
            total_r2 += r2.item()
            max_mae = max(max_mae, mae.item())
            total_samples += targets.size(0)  # Increment total sample count

    # Compute average metrics over the entire test set
    avg_mse = total_mse / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)
    avg_r2 = total_r2 / len(test_dataloader)

    # Output test results
    print(f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max MAE: {max_mae:.6f}, Test R²: {avg_r2:.4f}")
    print(f"Total inference time: {total_inference_time:.2f}s for {total_samples} samples")
    return {'MSE': avg_mse, 'MAE': avg_mae, 'Max MAE': max_mae, 'R2': avg_r2}


def load_and_test_model(model_path, test_dataloader, device):
    """Load a saved model and test it, returning the test results."""
    model = RegPointNet(args=config).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 2, 3])
    model.load_state_dict(torch.load(model_path))

    return test_model(model, test_dataloader, config)


from sklearn.manifold import TSNE
def save_features_incrementally(features, filename):
    """ Save features incrementally to avoid large memory overhead. """
    with open(filename, 'ab') as f:
        np.save(f, features)


def extract_features_and_outputs(model_path, dataloader, device, config):
    """Load a saved model and extract features and outputs from the specified DataLoader, saving them to files."""
    # Load the model configuration and setup
    model = RegPointNet(args=config).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    features_list = []
    outputs_list = []
    total_batches = len(dataloader)
    tsne_save_path = os.path.join('models', f'{config["exp_name"]}_train_tsne.npy')

    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device).squeeze()
            data = data.permute(0, 2, 1)
            output, features = model(data)
            #save_features_incrementally(features.cpu().numpy(), tsne_save_path)

            features_list.append(features.cpu().numpy())
            #outputs_list.append(output.cpu().numpy())
            # Progress update
            if i % 10 == 0:  # Adjust the frequency of messages according to your needs
                print(f"Processed {i + 1}/{total_batches} batches.")
    print("Saving Results")
    # Concatenate all batches to form a single array for features and outputs
    features_array = np.concatenate(features_list, axis=0)
    #outputs_array = np.concatenate(outputs_list, axis=0)

    # Apply t-SNE to the concatenated feature array
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)  # 2D t-SNE, modify parameters as needed
    tsne_results = tsne.fit_transform(features_array)

    # Save the extracted features, outputs, and t-SNE results to the disk
    tsne_save_path = os.path.join('models', f'{config["exp_name"]}_train_tsne.npy')

    np.save(tsne_save_path, tsne_results)
    print("t-SNE results saved to {tsne_save_path}")

    #return tsne_results


if __name__ == "__main__":
    setup_seed(config['seed'])

    # List of fractions of the training data to use
    train_fractions = [1.0]
    results = {}

    for frac in train_fractions:
        print(f"Training on {frac * 100}% of the training data")
        model = initialize_model(config).to(device)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            config['dataset_path'], config['aero_coeff'], config['subset_dir'],
            config['num_points'], config['batch_size'], train_frac=frac
        )

        best_model_path, final_model_path = train_and_evaluate(model, train_dataloader, val_dataloader, config)

        # Test the best model
        print("Testing the best model:")
        best_results = load_and_test_model(best_model_path, test_dataloader, device)

        # Test the final model
        print("Testing the final model:")
        final_results = load_and_test_model(final_model_path, test_dataloader, device)

        # Store results
        #results[f"{int(frac * 100)}%_best"] = best_results
        #results[f"{int(frac * 100)}%_final"] = final_results
        #best_model_path= '../DragPrediction_DrivAerNet_PointNet_r2_batchsize32_200epochs_100kpoints_tsne_NeurIPS_best_model.pth'
        #outputs, features = extract_features_and_outputs(best_model_path, train_dataloader, device, config)

    # Save the results to a CSV file
    #df_results = pd.DataFrame(results)
    #df_results.to_csv('model_training_results_PC_normalized.csv')
    #print("Results saved to model_training_results.csv")
