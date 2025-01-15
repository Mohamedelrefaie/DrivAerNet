#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

This script trains and evaluates graph-based neural networks, specifically DragGNN models, for predicting aerodynamic drag.
"""
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader, random_split, Subset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset, DataLoader as GeoDataLoader
from tqdm import tqdm
import numpy as np
from DeepSurrogate_models import DragGNN, DragGNN_XL, EnhancedDragGNN
from DrivAerNetDataset import DrivAerNetGNNDataset
import pandas as pd

# Configuration dictionary to hold hyperparameters and settings
config = {
    'cuda': True,
    'exp_name': 'DragPrediction_DrivAerNet_DragGNN_100epochs_NeurIPS',
    'seed': 1,
    'batch_size':1,
    'epochs': 100,
    'lr': 0.001,
    'optimizer': 'adam',
    'dataset_path': '../DrivAerNet_FEN_STLs',  # Update this with your dataset path
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

    # Instantiate the RegDGCNN model with the specified configuration parameters
    #model = DragGNN_XL().to(device)
    model = DragGNN_XL().to(device)
    # If CUDA is enabled and more than one GPU is available, wrap the model in a DataParallel module
    # to enable parallel computation across multiple GPUs. Specifically, use GPUs with IDs 0, 1, 2, and 3.
    if config['cuda'] and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

    # Return the initialized model
    return model


def get_dataloaders(dataset_path: str, aero_coeff: str, subset_dir: str,batch_size: int) -> tuple:
    """
    Prepare and return the training, validation, and test DataLoader objects.

    Args:
        dataset_path (str): The file path to the dataset directory containing the STL files.
        aero_coeff (str): The path to the CSV file with metadata for the models.
        subset_dir (str): The directory containing the subset files (train, val, test).
        num_points (int): The number of points to sample from each point cloud in the dataset.
        batch_size (int): The number of samples per batch to load.

    Returns:
        tuple: A tuple containing the training DataLoader, validation DataLoader, and test DataLoader.
    """

    # Initialize the full dataset
    full_dataset = DrivAerNetGNNDataset(root_dir=dataset_path, csv_file=aero_coeff)

    # Helper function to create subsets from IDs in text files
    def create_subset(dataset, ids_file):
        try:
            with open(os.path.join(subset_dir, ids_file), 'r') as file:
                subset_ids = file.read().split()
            # Filter the dataset DataFrame based on subset IDs
            subset_indices = dataset.data_frame[dataset.data_frame['Design'].isin(subset_ids)].index.tolist()
            return Subset(dataset, subset_indices)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")

    # Create each subset using the corresponding subset file
    train_dataset = create_subset(full_dataset, 'train_design_ids.txt')
    val_dataset = create_subset(full_dataset, 'val_design_ids.txt')
    test_dataset = create_subset(full_dataset, 'test_design_ids.txt')

    # Initialize DataLoaders for each subset
    train_dataloader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_dataloader = GeoDataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_dataloader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader

def train_and_evaluate(model: torch.nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, config: dict):
    """
    Train and evaluate the model using the provided dataloaders and configuration.

    Args:
        model (torch.nn.Module): The model to be trained and evaluated.
        train_dataloader (DataLoader): Dataloader for the training set.
        val_dataloader (DataLoader): Dataloader for the validation set.
        config (dict): Configuration dictionary containing training hyperparameters and settings.

    """
    train_losses, val_losses = [], []
    training_start_time = time.time()  # Start timing for training
    device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")

    # Initialize the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4) if config[
                                                                                          'optimizer'] == 'adam' else optim.SGD(
        model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)

    # Initialize the learning rate scheduler (ReduceLROnPlateau) to reduce the learning rate based on validation loss
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.1, verbose=True)

    best_mse = float('inf')  # Initialize the best MSE as infinity

    # Training loop over the specified number of epochs
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()  # Start timing for this epoch
        model.train()  # Set the model to training mode
        total_loss, total_r2 = 0, 0

        # Iterate over the training data
        for data in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Training]"):
            data = data.to(device)  # Move data to the gpu

            optimizer.zero_grad()
            outputs = model(data)
            loss = F.mse_loss(outputs.squeeze() , data.y)
            r2 = r2_score(outputs.squeeze() ,data.y)  # Compute R2 score

            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # Accumulate the loss
            total_r2 += r2.item()

        epoch_duration = time.time() - epoch_start_time
        # Calculate and print the average training loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.6f} Time: {epoch_duration:.2f}s")

        avg_r2 = total_r2 / len(train_dataloader)
        print(f"Average Training R²: {avg_r2:.4f}")

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss, val_r2 = 0, 0
        inference_times = []

        # No gradient computation needed during validation
        with torch.no_grad():
            # Iterate over the validation data
            for data in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Validation]"):
                inference_start_time = time.time()
                data = data.to(device)
                outputs = model(data)
                loss = F.mse_loss(outputs.squeeze(), data.y)
                val_loss += loss.item()
                r2 = r2_score(outputs.squeeze(), data.y)  # Compute R2 score
                val_r2 += r2.item()
                inference_duration = time.time() - inference_start_time
                inference_times.append(inference_duration)

        # Calculate and print the average validation loss for the epoch
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        avg_inference_time = sum(inference_times) / len(inference_times)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Avg Inference Time: {avg_inference_time:.4f}s")

        avg_val_r2 = val_r2 / len(val_dataloader)
        print(f"Average Validation R²: {avg_val_r2:.4f}")

        # Check if this is the best model based on MSE
        if avg_val_loss < best_mse:
            best_mse = avg_val_loss
            best_model_path = os.path.join('models', f'{config["exp_name"]}_best_model.pth')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with MSE: {best_mse:.6f} and R²: {avg_val_r2:.4f}")

        # Step the scheduler based on the validation loss
        scheduler.step(avg_val_loss)

    training_duration = time.time() - training_start_time
    print(f"Total training time: {training_duration:.2f}s")
    # Save the final model state to disk
    model_path = os.path.join('models', f'{config["exp_name"]}_final_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # Save losses for plotting
    np.save(os.path.join('models', f'{config["exp_name"]}_train_losses.npy'), np.array(train_losses))
    np.save(os.path.join('models', f'{config["exp_name"]}_val_losses.npy'), np.array(val_losses))

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
        for data in test_dataloader:
            start_time = time.time()  # Start time for inference

            data= data.to(device)
            outputs = model(data)

            end_time = time.time()  # End time for inference
            inference_time = end_time - start_time
            total_inference_time += inference_time  # Accumulate total inference time

            mse = F.mse_loss(outputs.squeeze(), data.y) #Mean Squared Error (MSE)
            mae = F.l1_loss(outputs.squeeze(), data.y) #Mean Absolute Error (MAE),
            r2 = r2_score(outputs.squeeze(), data.y) #R-squared

            # Accumulate metrics to compute averages later
            total_mse += mse.item()
            total_mae += mae.item()
            total_r2 += r2.item()
            max_mae = max(max_mae, mae.item())
            total_samples += data.y.size(0)  # Increment total sample count

    # Compute average metrics over the entire test set
    avg_mse = total_mse / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)
    avg_r2 = total_r2 / len(test_dataloader)

    # Output test results
    print(f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max MAE: {max_mae:.6f}, Test R²: {avg_r2:.4f}")
    print(f"Total inference time: {total_inference_time:.2f}s for {total_samples} samples")


def load_and_test_model(model_path, test_dataloader, device):
    """Load a saved model and test it."""
    model = DragGNN_XL().to(device)  # Initialize a new model instance
    #model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(torch.load(model_path))  # Load the saved weights

    test_model(model, test_dataloader, config)


if __name__ == "__main__":
    setup_seed(config['seed'])
    # Prepare data
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config['dataset_path'],config['aero_coeff'],
                                                          config['subset_dir'], config['batch_size'])

    # Initialize model
    model = DragGNN_XL().to(device)
    #model = initialize_model(config)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # Train and evaluate
    train_and_evaluate(model, train_dataloader, val_dataloader, config)

    # Load and test both the best and final models
    final_model_path = os.path.join('models', f'{config["exp_name"]}_final_model.pth')
    print("Testing the final model:")
    load_and_test_model(final_model_path, test_dataloader, device)

    best_model_path = os.path.join('models', f'{config["exp_name"]}_best_model.pth')
    print("Testing the best model:")
    load_and_test_model(best_model_path, test_dataloader, device)
