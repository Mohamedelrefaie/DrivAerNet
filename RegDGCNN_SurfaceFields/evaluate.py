# evaluate.py
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu

Evaluation script for pressure field prediction models on the DrivAerNet++ dataset.

This script handles the evaluation and visualization of prediction results from
trained RegDGCNN models on the test dataset.
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from data_loader import SurfacePressureDataset, PRESSURE_MEAN, PRESSURE_STD
from model_pressure import RegDGCNN_pressure
from utils import setup_logger, setup_seed, visualize_pressure_field, plot_error_distribution, calculate_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate pressure prediction models on DrivAerNet++')
    
    # Basic settings
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name for results folder')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    
    # Data settings
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--cache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points to sample')
    parser.add_argument('--sample_ids', type=str, help='Path to file with sample IDs to evaluate')
    
    # Model settings
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (for model initialization)')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions (for model initialization)')
    parser.add_argument('--k', type=int, default=40, help='Number of nearest neighbors (for model initialization)')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of output channels')
    
    # Visualization settings
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--num_vis_samples', type=int, default=5, help='Number of samples to visualize')
    
    return parser.parse_args()


def initialize_model(args, device):
    """
    Initialize and load the model.

    Args:
        args: Command line arguments
        device: PyTorch device to use

    Returns:
        Loaded model
    """
    # Convert args to dictionary ONLY for the model initialization
    args_dict = vars(args)

    # Use args_dict only for model initialization
    model = RegDGCNN_pressure(args_dict).to(device)

    # Use original args for everything else
    logging.info(f"Loading model from {args.model_checkpoint}")
    state_dict = torch.load(args.model_checkpoint, map_location=device)

    # Remove 'module.' prefix from state dict keys if loading a DDP model to a non-DDP model
    if list(state_dict.keys())[0].startswith('module.') and not hasattr(model, 'module'):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    return model

def prepare_dataset(args):
    """
    Prepare the dataset for evaluation.
    
    Args:
        args: Command line arguments
        
    Returns:
        Prepared dataset and sample indices
    """
    # Create dataset
    dataset = SurfacePressureDataset(
        root_dir=args.dataset_path,
        num_points=args.num_points,
        preprocess=False,  # We don't need to preprocess if using cached data
        cache_dir=args.cache_dir
    )
    
    # Determine which samples to evaluate
    if args.sample_ids:
        try:
            with open(args.sample_ids, 'r') as f:
                sample_ids = [id_.strip() for id_ in f.readlines()]
            
            # Filter to only include VTK files that match the sample IDs
            sample_files = [f for f in dataset.vtk_files if any(id_ in f for id_ in sample_ids)]
            sample_indices = [dataset.vtk_files.index(f) for f in sample_files]
            
            logging.info(f"Found {len(sample_indices)} samples matching the provided IDs")
        except Exception as e:
            logging.error(f"Error loading sample IDs: {e}")
            sample_indices = list(range(len(dataset)))
    else:
        # Use all samples
        sample_indices = list(range(len(dataset)))
        
        # If visualizing, limit to the specified number
        if args.visualize and args.num_vis_samples < len(sample_indices):
            sample_indices = sample_indices[:args.num_vis_samples]
    
    return dataset, sample_indices


def evaluate_model(model, dataset, sample_indices, args):
    """
    Evaluate the model on the selected samples and save raw prediction data.

    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        sample_indices: Indices of samples to evaluate
        args: Command line arguments

    Returns:
        Dictionary of evaluation results
    """
    model.eval()
    device = next(model.parameters()).device

    all_metrics = []
    results_dir = os.path.join('results', args.exp_name)
    os.makedirs(results_dir, exist_ok=True)

    # Directory for saving raw prediction data
    data_dir = os.path.join(results_dir, 'prediction_data')
    if args.visualize:
        os.makedirs(data_dir, exist_ok=True)

    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Evaluating samples"):
            # Get sample
            data, targets = dataset[idx]

            # Skip invalid samples
            if data is None or targets is None:
                logging.warning(f"Skipping invalid sample at index {idx}")
                continue

            # Prepare inputs
            data = data.squeeze(1).to(device)
            targets = targets.squeeze(1).to(device)
            normalized_targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

            # Forward pass
            outputs = model(data)
            normalized_outputs = outputs.squeeze(1)

            # Calculate metrics on normalized values
            batch_metrics = calculate_metrics(normalized_targets.cpu().numpy(), normalized_outputs.cpu().numpy())
            all_metrics.append(batch_metrics)

            # Denormalize for data saving
            outputs = normalized_outputs * PRESSURE_STD + PRESSURE_MEAN

            # Save raw prediction data if requested
            if args.visualize:
                vtk_file = dataset.vtk_files[idx]
                sample_name = os.path.basename(vtk_file).replace('.vtk', '')

                # Extract points from the data tensor - correct format for later visualization
                points = data.cpu().numpy().squeeze(0).transpose(1, 0)  # (3, 10000) -> (10000, 3)
                true_pressure_np = targets.cpu().numpy().squeeze()
                pred_pressure_np = outputs.cpu().numpy().squeeze()

                # Save raw data for later visualization
                output_data = {
                    'points': points,
                    'true_pressure': true_pressure_np,
                    'pred_pressure': pred_pressure_np,
                    'sample_name': sample_name,
                    'vtk_file': vtk_file,
                    'metrics': batch_metrics
                }

                # Save to npz file
                data_path = os.path.join(data_dir, f"{sample_name}_prediction_data.npz")
                np.savez(data_path, **output_data)
                logging.info(f"Saved raw prediction data to {data_path}")

                # Calculate error metrics
                error = np.abs(true_pressure_np - pred_pressure_np)
                max_error = np.max(error)
                mean_error = np.mean(error)
                std_error = np.std(error)

                # Log some basic error statistics
                logging.info(
                    f"Sample: {sample_name}, Max Error: {max_error:.6f}, Mean Error: {mean_error:.6f}, Std Error: {std_error:.6f}")

    # Aggregate metrics
    agg_metrics = {}
    for metric_name in all_metrics[0].keys():
        agg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
        agg_metrics[f"{metric_name}_std"] = np.std([m[metric_name] for m in all_metrics])

    # Save metrics
    metrics_file = os.path.join(results_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation Metrics for RegDGCNN\n")
        f.write(f"Model Checkpoint: {args.model_checkpoint}\n")
        f.write(f"Number of samples: {len(sample_indices)}\n\n")

        for metric_name, value in agg_metrics.items():
            f.write(f"{metric_name}: {value:.6f}\n")

    # Also save aggregated metrics as numpy file for easy loading
    np.savez(os.path.join(results_dir, 'aggregated_metrics.npz'), **agg_metrics)

    logging.info(f"Evaluation complete. Results saved to {results_dir}")
    logging.info(f"Raw prediction data saved to {data_dir}")
    return agg_metrics


def main():
    """Main function to run the evaluation."""
    args = parse_args()
    setup_seed(args.seed)
    
    # Set up logging
    results_dir = os.path.join('results', args.exp_name)
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, 'evaluation.log')
    setup_logger(log_file)
    
    logging.info(f"Starting evaluation of RegDGCNN model")
    logging.info(f"Arguments: {args}")
    
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Initialize model
    model = initialize_model(args, device)
    model.eval()
    
    # Prepare dataset
    dataset, sample_indices = prepare_dataset(args)
    
    # Evaluate model
    metrics = evaluate_model(model, dataset, sample_indices, args)
    
    # Log results
    logging.info("Evaluation Results:")
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.6f}")


if __name__ == "__main__":
    main()
