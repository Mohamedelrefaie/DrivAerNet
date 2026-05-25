# CarMRI: Slice-based Surrogate Model for Drag Coefficient Prediction

## Award

**Miltos Petridis Memorial Trophy — Best Student Application Paper** at AI-2025, the 45th SGAI International Conference on Artificial Intelligence, BCS, Cambridge University (December 2025)

## Model Architecture

CarMRI predicts the aerodynamic drag coefficient (Cd) from 3D vehicle point clouds using a sequential, slice-based approach inspired by how MRI/CT scans represent 3D anatomical structures. The architecture (PointNet2D + BiLSTM) has three main components:

1. **PointNet2D (Slice-Level Feature Extractor):** Each 3D point cloud (~100k points) is sliced into S=80 cross-sectional 2D slices along the streamwise (X) axis. Each slice is processed independently by a lightweight 2D adaptation of PointNet using three 1D convolutional layers (channels: 2 -> 32 -> 64 -> 256) with ReLU activations, followed by global max-pooling to produce a 256-dimensional embedding per slice.

2. **Bi-Directional LSTM (Sequence Model):** The sequence of 80 slice embeddings is processed by a 2-layer Bi-LSTM (hidden dimension = 256 per direction) to capture longitudinal geometric dependencies in both flow directions. The final hidden states from both directions are concatenated into a 512-dimensional car-level embedding.

3. **MLP (Regression Head):** A 3-layer fully connected network (512 -> 256 -> 64 -> 1) with ReLU activations and dropout (0.3) regresses the scalar Cd value.

## Implementation Specifics

- **Input Representation:** Each car is represented as a tensor of shape (80, 6500, 2) — 80 slices, each zero-padded to 6500 points with (y, z) coordinates.
- **Framework:** PyTorch
- **Total Parameters:** ~2.80 million

## Training Configuration and Hyperparameters

- **Dataset:** DrivAerNet++ (7,713 samples after filtering: 5,398 train / 1,157 validation / 1,158 test)
- **Data Splits:** Official DrivAerNet++ train/validation/test splits
- **Loss Function:** Smooth L1 Loss (Huber Loss, beta=1.0)
- **Optimizer:** Adam (learning rate = 1e-4)
- **Batch Size:** 4
- **Epochs:** 100 (best model selected at epoch 68 based on highest validation R²)
- **Hardware:** Single NVIDIA RTX 4060 GPU (5.9 GB VRAM usage)
- **Training Time:** ~21 hours

## Results

| Metric | Value |
|--------|-------|
| MSE | 6.60 x 10⁻⁵ |
| MAE | 6.111 x 10⁻³ |
| Max MAE | 4.50 x 10⁻² |
| R² | 0.9525 |
| Inference Latency | 0.025 s/sample |
| Parameters | 2.80M |

## Link to Paper

- **Springer Nature LNCS:** [https://doi.org/10.1007/978-3-032-11442-6_5](https://doi.org/10.1007/978-3-032-11442-6_5)
- **arXiv:** [https://arxiv.org/abs/2601.02112](https://arxiv.org/abs/2601.02112)

## Link to Code and Model Weights

- **Repository:** https://github.com/Adarsh-Roy/cd_prediction
- **Training Code:** https://github.com/Adarsh-Roy/cd_prediction/tree/main/training
- **Trained Model Weights:** https://github.com/Adarsh-Roy/cd_prediction/tree/main/models

## Authors

- Utkarsh Singh (Delhi Technological University)
- Adarsh Roy (Indian Institute of Technology, Hauz Khas)
- Absaar Ali (Delhi Technological University)
