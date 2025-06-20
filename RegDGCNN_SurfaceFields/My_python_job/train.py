# train.py
import os
import torch
import argparse

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

def main():
    """ main function to parse arguments and start training."""
    args = parse_args() 

    # Set visible GPUS
    gpu_list = args.gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # Count number of GPUs to use
    world_size = len(gpu_list.split(','))
    print(f"world_size = {world_size}")

if __name__=="__main__":
    main()
