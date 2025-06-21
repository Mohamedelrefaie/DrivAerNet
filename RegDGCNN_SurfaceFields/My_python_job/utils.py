# utils.py
"""

@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu

Utility functions for the DrivAerNet pressure prediction project.

This module provides helper functions for logging, random seed setup,
visualization, and other common operations.
"""

import os
import random
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from matplotlib import cm
import pyvista as pv
from data_loader import PRESSURE_MEAN, PRESSURE_STD


def setup_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: The random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file=None, level=logging.INFO):
    """
    Set up the logger for the application.
    
    Args:
        log_file: Path to the log file
        level: Logging level
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)


def visualize_pressure_field(points, true_pressure, pred_pressure, output_path):
    """
    Visualize the true and predicted pressure fields on a 3D model.

    Args:
        points: 3D point cloud coordinates
        true_pressure: Ground truth pressure values
        pred_pressure: Predicted pressure values
        output_path: Path to save the visualization
    """
    # Reshape the points to be (num_points, 3)
    # The issue is points has shape (3, num_points)
    if points.shape[0] == 3 and points.ndim == 2:
        # Transpose to get (num_points, 3)
        points = points.T

    # Make sure pressure values are 1D arrays
    if true_pressure.ndim > 1:
        true_pressure = true_pressure.squeeze()
    if pred_pressure.ndim > 1:
        pred_pressure = pred_pressure.squeeze()

    # Denormalize pressure values if needed
    # true_pressure = true_pressure * PRESSURE_STD + PRESSURE_MEAN
    # pred_pressure = pred_pressure * PRESSURE_STD + PRESSURE_MEAN

    # Create PyVista point clouds
    true_cloud = pv.PolyData(points)
    true_cloud.point_data['pressure'] = true_pressure

    pred_cloud = pv.PolyData(points)
    pred_cloud.point_data['pressure'] = pred_pressure

    # Create PyVista plotter
    plotter = pv.Plotter(shape=(1, 2), off_screen=True)

    # Plot true pressure
    plotter.subplot(0, 0)
    plotter.add_text("True Pressure", font_size=16)
    plotter.add_mesh(true_cloud, scalars='pressure', cmap='jet', point_size=5)

    # Plot predicted pressure
    plotter.subplot(0, 1)
    plotter.add_text("Predicted Pressure", font_size=16)
    plotter.add_mesh(pred_cloud, scalars='pressure', cmap='jet', point_size=5)

    # Save figure
    plotter.screenshot(output_path)
    plotter.close()

def plot_error_distribution(true_pressure, pred_pressure, output_path):
    """
    Plot the distribution of prediction errors.
    
    Args:
        true_pressure: Ground truth pressure values
        pred_pressure: Predicted pressure values
        output_path: Path to save the plot
    """
    # Calculate absolute errors
    errors = np.abs(true_pressure - pred_pressure)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()


def save_training_curve(train_losses, val_losses, output_path):
    """
    Save a plot of training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', linestyle='-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()


def calculate_metrics(true_values, predicted_values):
    """
    Calculate various evaluation metrics.
    
    Args:
        true_values: Ground truth values
        predicted_values: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if tensors
    if torch.is_tensor(true_values):
        true_values = true_values.cpu().numpy()
    if torch.is_tensor(predicted_values):
        predicted_values = predicted_values.cpu().numpy()
    
    # Mean Squared Error
    mse = np.mean((true_values - predicted_values) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(true_values - predicted_values))
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Maximum Absolute Error
    max_error = np.max(np.abs(true_values - predicted_values))
    
    # Relative L2 Error (normalized)
    rel_l2 = np.mean(np.linalg.norm(true_values - predicted_values, axis=0) / 
                     np.linalg.norm(true_values, axis=0))
    
    # Relative L1 Error (normalized)
    rel_l1 = np.mean(np.sum(np.abs(true_values - predicted_values), axis=0) / 
                     np.sum(np.abs(true_values), axis=0))
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Max_Error': max_error,
        'Rel_L2': rel_l2,
        'Rel_L1': rel_l1
    }
