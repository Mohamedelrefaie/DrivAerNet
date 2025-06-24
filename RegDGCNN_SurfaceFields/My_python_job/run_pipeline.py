#!/usr/bin/env python3
# run_pipeline.py

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from   torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from   tqdm import tqdm
import logging

# Import modules
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from model_pressure import RegDGCNN_pressure
from utils import setup_logger, setup_seed

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
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpus', type=str, default='0', help='GPUs to use (comma-separated)')

    # Model settings
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--k', type=int, default=40, help='Number of nearest neighbors')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of output channels')


    return parser.parse_args()

def train_model(args):
    logging.info("Starting model training...")

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
        "--num_workers", str(args.num_workers),
        "--dropout", str(args.dropout),
        "--emb_dims", str(args.emb_dims),
        "--k", str(args.k),
        "--output_channels", str(args.output_channels),
        "--seed", str(args.seed)
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
    print(f"Model training completed ")
    logging.info(f"Model training completed in {elapsed_time:.2f} seconds")
    return True

def main():
    """ main function to run the complete pipeline. """
    args = parse_args()

    # Set up logging
    timestamp = datatime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", args_exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{timestamp.log}")
#    setup_logger(log_file)


if __name__=="__main__":
    exit(main())

