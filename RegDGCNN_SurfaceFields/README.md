# RegDGCNN for Surface Pressure Field Prediction

This folder contains code for predicting pressure fields on 3D car models from the DrivAerNet++ dataset using the RegDGCNN model (Deep Graph Convolutional Neural Network for Regression).

## Overview

The DrivAerNet++ dataset contains 3D car models with corresponding pressure field data from CFD simulations. This code implements:

- Preprocessing of VTK mesh files into point cloud data
- RegDGCNN model for pressure field prediction
- Multi-GPU distributed training
- Comprehensive evaluation metrics

## Setup

### Requirements

- Python 3.7+
- PyTorch 1.8+
- PyVista
- CUDA-capable GPU(s)

### Installation

```bash
# Clone the repository
git clone https://github.com/Mohamedelrefaie/DrivAerNet.git
cd RegDGCNN_SurfaceFields

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Complete Pipeline

To run the complete pipeline (preprocessing, training, evaluation):

```bash
python run_pipeline.py \
    --exp_name "DrivAerNet_Pressure" \
    --dataset_path "/path/to/PressureVTK" \
    --subset_dir "/path/to/train_val_test_splits" \
    --cache_dir "/path/to/cached_data" \
    --num_points 10000 \
    --batch_size 12 \
    --epochs 150 \
    --gpus "0,1,2,3"
```

### Individual Steps

#### Data Preprocessing

```bash
python run_pipeline.py \
    --stages preprocess \
    --exp_name "DrivAerNet_Pressure" \
    --dataset_path "/path/to/PressureVTK" \
    --cache_dir "/path/to/cached_data" \
    --num_points 10000
```

#### Model Training

```bash
python run_pipeline.py \
    --stages train \
    --exp_name "DrivAerNet_Pressure" \
    --dataset_path "/path/to/PressureVTK" \
    --subset_dir "/path/to/train_val_test_splits" \
    --cache_dir "/path/to/cached_data" \
    --num_points 10000 \
    --batch_size 12 \
    --epochs 150 \
    --gpus "0,1,2,3"
```

#### Model Evaluation

```bash
python run_pipeline.py \
    --stages evaluate \
    --exp_name "DrivAerNet_Pressure" \
    --dataset_path "/path/to/PressureVTK" \
    --subset_dir "/path/to/train_val_test_splits" \
    --cache_dir "/path/to/cached_data" \
    --num_points 10000 \
    --num_eval_samples 5 \
    --gpus "0"
```

### Testing a Single VTK File

To test and visualize results for a specific car model:

```bash
python test_single_vtk.py \
    --model_checkpoint "experiments/DrivAerNet_Pressure/best_model.pth" \
    --vtk_file "/path/to/PressureVTK/car_model.vtk" \
    --output_dir "visualizations" \
    --num_points 10000 \
    --k 40 \
    --emb_dims 1024 \
    --dropout 0.4
```

## Directory Structure

```
.
├── data_loader.py         # Dataset loading and preprocessing
├── model_pressure.py      # RegDGCNN model implementation
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── test_single_vtk.py     # Test on a single VTK file
├── utils.py               # Utility functions
├── run_pipeline.py        # Pipeline script for end-to-end execution
├── experiments/           # Trained models and logs
└── results/               # Evaluation results and data
```

## Model Architecture

The RegDGCNN architecture uses dynamic graph convolutions to process point cloud data, capturing both local and global geometric features. The model learns to predict pressure values at each point on the car surface.

## Citation

If you use this code in your research, please cite the original paper:

```
@inproceedings{NEURIPS2024_013cf29a,
 author = {Elrefaie, Mohamed and Morar, Florin and Dai, Angela and Ahmed, Faez},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {499--536},
 publisher = {Curran Associates, Inc.},
 title = {DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/013cf29a9e68e4411d0593040a8a1eb3-Paper-Datasets_and_Benchmarks_Track.pdf},
 volume = {37},
 year = {2024}
}
```

## License

Code is distributed under the MIT License, while the DrivAerNet/DrivAerNet++ dataset is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

## Contact

Mohamed Elrefaie - mohamed.elrefaie [at] mit.edu
