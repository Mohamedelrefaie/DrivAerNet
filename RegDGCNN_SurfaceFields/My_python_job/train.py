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

def train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for data, targets in tqdm(train_dataloader, desc="[Training]"):
        data    = data.squeeze(1).to(local_rank)
        tragets = targets.squeeze(1).to(local_rank)
        targets = (targets - PRESSURE_MEAN) / PRESSURE_STD
        targets = targets.to(local_rank)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(1), targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)

def validate(model, val_dataloader, criterion, local_rank):
    """ Validate the model"""
    total_loss = 0

    with torch.no_grad():
        for data, targets in tqdm(val_dataloader, desc="[Validation]"):
            data    = data.squeeze(1).to(local_rank)
            targets = targets.squeeze(1).to(local_rank)
            targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

            outputs     = model(data)
            loss        = criterion(outputs.squeeze(1), targets)
            total_loss += loss.item()
    return total_loss / len(val_dataloader)

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

    # Initialize model
    model = initialize_model(args, local_rank)

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params}")

    # Prepare DataLoaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        args.dataset_path, args.subset_dir, args.num_points,
        args.batch_size, world_size, rank, args.cache_dir,
        args.num_workers
    )


    # Log dataset info
    if local_rank == 0:
        logging.info(
            f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches")

    # Set up criterion, optimizer, and scheduler
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

    best_model_path  = os.path.join('experiments', args.exp_name, 'best_model_pth')
    final_model_path = os.path.join('experiments', args.exp_name, 'final_model_pth')

    # Check if test_only and model exists
    if args.test_only and os.path.exists(best_model_path):
        if local_rank == 0:
            logging.info("Loading best model for testing only")
            print("Testing the best model:")
        model.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{local_rank}'))
        test_model(model, test_dataloader, criterion, local_rank, os.path.join('experiments', args.exp_name))
        dist.destroy_process_group()
        return

    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    if local_rank == 0:
        logging.info(f"Staring training for {args.epochs} epochs")

    # Training loop
    for epoch in range(args.epochs):
        # Set epoch for the DistributedSampler
        train_dataloader.sampler.set_epoch(epoch)

        # Training
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank)

        # Validation
        val_loss = validate(model, val_dataloader, criterion, local_rank)

        # Record losses
        if local_rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            logging.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"New best model saved with Val Loss: {best_val_loss:.6f}")

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Save progress rate scheduler
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
                plt.plot(range(1, epoch + 2), val_losses,   label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title(f'Training Progress - RegDGCNN')
                plt.savefig(os.path.join('experiments', args.exp_name, f'training_progress_epoch_{epoch+1}.png'))
                plt.close()

    # Save final model
    if local_rank == 0:
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

    # Make sure all processes sync up before testing
    dist.barrier()

    # Test the best model
    if local_rank == 0:
        log.info("Testing the final model")
        model.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{localhost}'))
        #test_model(model, test_dataloader, criterion, local_rank, os.path.join('experiments', args.exp_name))

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

'''
    # Checkout the DataLoader object
    logging.info(f"Type of train_dataloader: {type(train_dataloader)}")
    logging.info(f"Number of train_dataloader: {len(train_dataloader)}")
    #logging.info(f"List all methods and attributs of Dataloader: {dir(train_dataloader)}")
    logging.info(f"We can access the internal conetnt by dataloader: ")
    for ii, (points, pressure)in enumerate(train_dataloader):
        logging.info(f"Batch: {ii}")
        logging.info(f"Batch.points.shape: {points.shape}") # [2, 1, 3, 10000]

        sample_0 = points[0]                    # [1, 3, 10000]
        sample_1 = points[1]                    # [1, 3, 10000]
        logging.info(f"points_sample_0.shape: {sample_0.shape}")

        sample_0 = sample_0.squeeze(0)          # [3, 10000]
        sample_1 = sample_1.squeeze(0)          # [3, 10000]
        logging.info(f"points_sample_0.shape: {sample_0.shape}")

        x0 = sample_0[0]
        x1 = sample_1[0]
        logging.info(f"The first 10 points in x_coor for the sample_0: {x0[:10]}")
        logging.info(f"The first 10 points in x_coor for the sample_1: {x1[:10]}")

        logging.info(f"Batch.Pressure.shape: {pressure.shape}") #[2, 1, 10000]

        sample_0 = pressure[0]                    # [1, 10000]
        sample_1 = pressure[1]                    # [1, 10000]
        logging.info(f"pressure_sample_0.shape: {sample_0.shape}")

        sample_0 = sample_0.squeeze(0)          # [10000]
        sample_1 = sample_1.squeeze(0)          # [10000]
        logging.info(f"pressure_sample_0.shape: {sample_0.shape}")

        logging.info(f"The first 10 points pressure for the sample_0: {sample_0[:10]}")
        logging.info(f"The first 10 points pressure for the sample_1: {sample_1[:10]}")


    # Checkout the torch.utils.data.subset object
    train_subset = train_dataloader.dataset
    logging.info(f"Type of train_subset: {type(train_subset)}")
    logging.info(f"Number of samples of train_subset : {len(train_subset)}")
    logging.info(f"Subset indices: {train_subset.indices[:5]}")
    logging.info(f"List the train_subset vtk files:")
    for ii, idx in enumerate(train_subset.indices):
        vtk_file = train_subset.dataset.vtk_files[idx]
        logging.info(f"{ii:3d}: {vtk_file}")
    #logging.info(f"List all methods and attributs of subset: {dir(dataset)})")


    # Checkout the full_dataset i.e. all .vtk files
    full_dataset = train_subset.dataset
    logging.info(f"Type of full_dataset: {type(full_dataset)}")
    logging.info(f"Number of samples of full_dataset: {len(full_dataset.vtk_files)}")
    for f, ii in enumerate(full_dataset.vtk_files):
        logging.info(f"  {ii: >2}: {f}")
'''
