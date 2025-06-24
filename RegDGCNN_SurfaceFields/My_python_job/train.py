# train.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Import modules
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from model_pressure import RegDGCNN_pressure
from utils import setup_logger, setup_seed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train pressure prediction models on DrivAerNet++')

    # Basic settings
    parser.add_argument('--exp_name', type=str, default='PressurePrediction', help='Experiment name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # Data settings
    parser.add_argument('--dataset_path', type=str,  help='Path to dataset')
    parser.add_argument('--subset_dir', type=str, help='Path to train/val/test splits')
    parser.add_argument('--cache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points to sample')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--test_only', action='store_true', help='Only test the model, no training')
    parser.add_argument('--gpus', type=str, default='0', help='GPUs to use (comma-separated)')

    # Model settings
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--k', type=int, default=40, help='Number of nearest neighbors')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of output channels')

    return parser.parse_args()

def initialize_model(args, local_rank):
    """ Initialize and return the RegDGCN model. """
    args = vars(args)
    model = RegDGCNN_pressure(args).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            output_device=local_rank
    )
    return model

def train_and_evaluate(rank, world_size, args):
    """ main function for Distributed training and evaluation. """
    setup_seed(args.seed)

    # Initialize process group for DDP
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    local_rank = rank
    torch.cuda.set_device(local_rank)

    # Set up logging (only on rank 0)
    if local_rank == 0:
        exp_dir = os.path.join('experiments', args.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        log_file = os.path.join(exp_dir, 'training.log')
        setup_logger(log_file)
        logging.info(f"args.exp_name : {args.exp_name}")
        logging.info(f"Arguments: {args}")
        logging.info(f"Starting training with {world_size} GPUs")

        # Checkout .npz file
        data = np.load("./Cache_data/N_S_WWS_WM_001.npz")
        logging.info("Key in the .npz file", data.files)
        for key in data.files:
            logging.info(f"{key}: shape = {data[key].shape}, dtype = {data[key].dtype}")

    # Initialize model
    model = initialize_model(args, local_rank)

#    if local_rank == 0:
#        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#        logging.info(f"Total trainable parameters: {total_params}")
#        print(f"Total trainable parameters: {total_params}")
#
#    # Prepare DataLoaders
#    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
#        args.dataset_path, args.subset_dir, args.num_points,
#        args.batch_size, world_size, rank, args.cache_dir,
#        args.num_workers
#    )
#
#    # Log dataset info
#    if local_rank == 0:
#        logging.info(
#            f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches")
#        print(
#            f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches")

    # Clean up
    dist.destroy_process_group()
def main():
    """ main function to parse arguments and start training."""
    args = parse_args()

    # Set the master address and port for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Set visible GPUS
    gpu_list = args.gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # Count number of GPUs to use
    world_size = len(gpu_list.split(','))

    # Create experiment directory
    exp_dir = os.path.join('experiments', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)


    # Start distributed training
    mp.spawn(train_and_evaluate, args=(world_size, args), nprocs=world_size, join=True)


if __name__=="__main__":
    main()

