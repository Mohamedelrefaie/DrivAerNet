#!/usr/bin/env python3
# run_pipeline.py

import os
import argparse
import subprocess
import logging
import time
import pprint
from datetime import datetime
from utils import setup_logger
from colorama import Fore, Style



#logging.basicConfig(
#    level=logging.INFO,  # <-- This enables logging.info()
#    format='[%(asctime)s] %(levelname)s: %(message)s',
#    datefmt='%H:%M:%S'
#)

def parse_args():

    parser = argparse.ArgumentParser(description="Test")

    # Pipeline control
    parser.add_argument('--stages', type=str, default='all',
                        choices=['preprocess', 'train', 'evaluate', 'all'],
                        help='Pipeline stages to run')

    # Basic settings
    parser.add_argument('--exp_name', type=str, required=True, help="Test")
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # Data settings
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--subset_dir', type=str, help='Path to train/val/test splits')
    parser.add_argument('--cache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points to sample')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_only', type=int, default=0, help='Only test the model, no training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpus', type=str, default='0', help='GPUs to use (comma-separated)')

    # Model settings
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--k', type=int, default=40, help='Number of nearest neighbors')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of output channels')

    # Evaluation settings
    parser.add_argument('--num_eval_samples', type=int, default=5, help='Number of samples to evaluate in detail')

    return parser.parse_args()

def preprocess_data(args):

    """
    Preprocess the dataset to create cached point cloud data.

    Args:
        True if preprocessing was successful, False otherwise
    """

    logging.info("**************************Starting data preprocessing...")

    # Create cache directory if it doesn't exist
    cache_dir = args.cache_dir or os.path.join(args.dataset_path, "processed_data")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Import required modules for preprocessing
        from data_loader import SurfacePressureDataset

        # Create the dataset with preprocessing enabled
        dataset = SurfacePressureDataset(
            root_dir = args.dataset_path,
            num_points = args.num_points,
            preprocess = True,
            cache_dir = cache_dir
            )

        # Process all files
        logging.info(f"Processing {len(dataset.vtk_files)} VTK files with {args.num_points} points per sample")
        for ii, vtk_file in enumerate(dataset.vtk_files):
            logging.info(f"Processing file {ii+1} / {len(dataset.vtk_files)}: {os.path.basename(vtk_file)}")
            _ = dataset[ii] # This will trigger preprocessing and caching

        logging.info(f"{Fore.MAGENTA}Data preprocessing complete. Cache data saved to {cache_dir}{Style.RESET_ALL}")
        return True
    except Exception as e:
        logging.error(f"Preprocessing failed with error: {e}")
        return False

def train_model(args):
    logging.info("*************************Starting model training...")

    # Prepare command for training script
    cmd = [
        "python", "train.py",
        "--exp_name", args.exp_name,
        "--dataset_path", args.dataset_path,
        "--subset_dir", args.subset_dir,
        "--num_points", str(args.num_points),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--dropout", str(args.dropout),
        "--emb_dims", str(args.emb_dims),
        "--k", str(args.k),
        "--output_channels", str(args.output_channels),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        "--test_only", str(args.test_only)
    ]

    if args.cache_dir:
        cmd.extend(["--cache_dir", args.cache_dir])

    # Set up environment variables for distributed training
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Run the training script
    start_time = time.time()
    process = subprocess.Popen(cmd, env=env)
    process.wait()

    if process.returncode != 0:
        logging.error("Training failed!")
        return False

    elapsed_time = time.time() - start_time
    logging.info(f"**********************Model training completed ")
    logging.info(f"Model training completed in {elapsed_time:.2f} seconds")
    return True

def evaluate_model(args):
    """
    Evaluatie the trained model using the evaluate.py script.

    Args:
        args: Command line arguments

    Returns:
        True if evaluation was successful, False otherwise
    """
    logging.info("*************************starting model evaluation...")

    # Path to the trained model
    model_checkpoint = os.path.join("experiments", args.exp_name, "best_model_pth")

    if not os.path.exists(model_checkpoint):
        logging.error(f"Model checkpoint not found at {model_checkpoint}")
        return False

    # Prepare command for evaluation script
    cmd = [
        "python", "evaluate.py",
        "--exp_name", args.exp_name,
        "--model_checkpoint", model_checkpoint,
        "--dataset_path", args.dataset_path,
        "--num_points", str(args.num_points),
        "--num_vis_samples", str(args.num_eval_samples),
        "--visualize",  # This will now save raw data instead of generating plots
        "--dropout", str(args.dropout),
        "--emb_dims", str(args.emb_dims),
        "--k", str(args.k),
        "--output_channels", str(args.output_channels),
        "--seed", str(args.seed)
    ]

    if args.cache_dir:
        cmd.extend(["--cache_dir", args.cache_dir])

    # Run the evaluation script
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus.split(',')[0]  # Use just the first GPU for evaluation

    process = subprocess.Popen(cmd, env=env)
    process.wait()

    if process.returncode != 0:
        logging.error("Evaluation failed!")
        return False

    logging.info("Model evaluation complete")
    return True


def main():
    """ main function to run the complete pipeline. """
    args = parse_args()

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    setup_logger(log_file)

    logging.info(f"Starting DrivAerNet pipeline - Experiment: {args.exp_name}")
    logging.info(f"Arguments:\n" + pprint.pformat(vars(args), indent=2))

    # Execute the selected pipeline stages
    stages = args.stages.split(',') if ',' in args.stages else [args.stages]
    if 'all' in stages:
        stages = ['preporcess', 'train', 'evaluate']

    results = {}

    # Preprocess data if requested
    if 'preprocess' in stages:
        results['preprocess'] = preprocess_data(args)
    else:
        # Skip preprocessing but mark as successful to allow training to proceed
        results['preprocess'] = True
        logging.info("Preprocessing stage skipped.")

    # Train model if requested and preprocessing succeeded
    if 'train' in stages and results['preprocess']:
        results['train'] = train_model(args)
    else:
        # If training not requested, mark as successful for evaluation to proceed
        if 'train' not in stages:
            results['train'] = True
            logging.info("Training stage skipped.")

    # Evaluate model if requested and training succeeded
    if 'evaluate' in stages and results.get('train', False):
        results['evaluate'] = evaluate_model(args)
    else:
        # If evaluation not requested, mark as true for final success check
        if 'evaluate' not in stages:
            results['evaluate'] = True
            logging.info("Evaluation stage skipped.")

    # Print Summary
    logging.info("Pipleline execution complete.")
    logging.info("Results summary: ")
    for stage, success in results.items():
        status = "Success" if success else "Failed"
        logging.info(f" {stage}: {status}")

    # Check if experiment was successful overall
#    overall_success = all(results.values())
#    loggiing.info(f"Overall status: {'Success' if overall_success else 'Failed'}")
#    return 0 if overall_success else 1


if __name__=="__main__":
    exit(main())

