#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

This module is used to define both point-cloud based and graph-based models, including RegDGCNN, PointNet, and several Graph Neural Network (GNN) models
for the task of surrogate modeling of the aerodynamic drag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import math
import numpy as np
import trimesh
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, JumpingKnowledge
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import BatchNorm

def knn(x, k):
    """
    Computes the k-nearest neighbors for each point in x.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of nearest neighbors to find.

    Returns:
        torch.Tensor: Indices of the k-nearest neighbors for each point, shape (batch_size, num_points, k).
    """
    # Calculate pairwise distance, shape (batch_size, num_points, num_points)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # Retrieve the indices of the k nearest neighbors
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Constructs local graph features for each point by finding its k-nearest neighbors and
    concatenating the relative position vectors.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of neighbors to consider for graph construction.
        idx (torch.Tensor, optional): Precomputed k-nearest neighbor indices.

    Returns:
        torch.Tensor: The constructed graph features of shape (batch_size, 2*num_dims, num_points, k).
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    # Compute k-nearest neighbors if not provided
    if idx is None:
        idx = knn(x, k=k)

    # Prepare indices for gathering
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()

    # Gather neighbors for each point to construct local regions
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    # Expand x to match the dimensions for broadcasting subtraction
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # Concatenate the original point features with the relative positions to form the graph features
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class RegDGCNN(nn.Module):
    """
    Deep Graph Convolutional Neural Network for Regression Tasks (RegDGCNN) for processing 3D point cloud data.

    This network architecture extracts hierarchical features from point clouds using graph-based convolutions,
    enabling effective learning of spatial structures.
    """

    def __init__(self, args, output_channels=1):
        """
        Initializes the RegDGCNN model with specified configurations.

        Args:
            args (dict): Configuration parameters including 'k' for the number of neighbors, 'emb_dims' for embedding
            dimensions, and 'dropout' rate.
            output_channels (int): Number of output channels (e.g., for drag prediction, this is 1).
        """
        super(RegDGCNN, self).__init__()
        self.args = args
        self.k = args['k']  # Number of nearest neighbors

        # Batch normalization layers to stabilize and accelerate training
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        self.bn5 = nn.BatchNorm1d(args['emb_dims'])

        # EdgeConv layers: Convolutional layers leveraging local neighborhood information
        self.conv1 = nn.Sequential(nn.Conv2d(6, 256, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(256 * 2, 512, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(512 * 2, 1024, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(2304, args['emb_dims'], kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        # Fully connected layers to interpret the extracted features and make predictions
        self.linear1 = nn.Linear(args['emb_dims']*2, 128, bias=False)
        self.bn6 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout(p=args['dropout'])

        self.linear2 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(p=args['dropout'])

        self.linear3 = nn.Linear(64, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.dp3 = nn.Dropout(p=args['dropout'])

        self.linear4 = nn.Linear(32, 16)
        self.bn9 = nn.BatchNorm1d(16)
        self.dp4 = nn.Dropout(p=args['dropout'])

        self.linear5 = nn.Linear(16, output_channels)  # The final output layer

    def forward(self, x):
        """
        Forward pass of the model to process input data and predict outputs.

        Args:
            x (torch.Tensor): Input tensor representing a batch of point clouds.

        Returns:
            torch.Tensor: Model predictions for the input batch.
        """
        batch_size = x.size(0)

        # Extract graph features and apply EdgeConv blocks
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 256, num_points, k)

        # Global max pooling
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # Repeat the process for subsequent EdgeConv blocks
        x = get_graph_feature(x1, k=self.k)   # (batch_size, 256, num_points) -> (batch_size, 256*2, num_points, k)
        x = self.conv2(x)                     # (batch_size, 256*2, num_points, k) -> (batch_size, 512, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)

        x = get_graph_feature(x2, k=self.k)   # (batch_size, 512, num_points) -> (batch_size, 512*2, num_points, k)
        x = self.conv3(x)                     # (batch_size, 512*2, num_points, k) -> (batch_size, 512, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)

        x = get_graph_feature(x3, k=self.k)   # (batch_size, 512, num_points) -> (batch_size, 512*2, num_points, k)
        x = self.conv4(x)                     # (batch_size, 512*2, num_points, k) -> (batch_size, 1024, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points, k) -> (batch_size, 1024, num_points)

        # Concatenate features from all EdgeConv blocks
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 256+512+512+1024, num_points)

        # Apply the final convolutional block
        x = self.conv5(x)  # (batch_size, 256+512+512+1024, num_points) -> (batch_size, emb_dims, num_points)
        # Combine global max and average pooling features
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)   # (batch_size, emb_dims*2)

        # Process features through fully connected layers with dropout and batch normalization
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 128)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 128) -> (batch_size, 64)
        x = self.dp2(x)
        x = F.leaky_relu(self.bn8(self.linear3(x)), negative_slope=0.2)  # (batch_size, 64) -> (batch_size, 32)
        x = self.dp3(x)
        x = F.leaky_relu(self.bn9(self.linear4(x)), negative_slope=0.2)  # (batch_size, 32) -> (batch_size, 16)
        x = self.dp4(x)

        # Final linear layer to produce the output
        x = self.linear5(x)                                              # (batch_size, 16) -> (batch_size, 1)

        return x


class RegPointNet(nn.Module):
    """
    PointNet-based regression model for 3D point cloud data.

    Args:
        args (dict): Configuration parameters including 'emb_dims' for embedding dimensions and 'dropout' rate.

    Methods:
        forward(x): Forward pass through the network.
    """
    def __init__(self, args):
        """
        Initialize the RegPointNet model for regression tasks with enhanced complexity,
        including additional layers and residual connections.

        Parameters:
            emb_dims (int): Dimensionality of the embedding space.
            dropout (float): Dropout probability.
        """
        super(RegPointNet, self).__init__()
        self.args = args

        # Convolutional layers
        self.conv1 = nn.Conv1d(3, 512, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
        self.conv6 = nn.Conv1d(1024, args['emb_dims'], kernel_size=1, bias=False)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(args['emb_dims'])

        # Dropout layers
        self.dropout_conv = nn.Dropout(p=args['dropout'])
        self.dropout_linear = nn.Dropout(p=args['dropout'])

        # Residual connection layer
        self.conv_shortcut = nn.Conv1d(3, args['emb_dims'], kernel_size=1, bias=False)
        self.bn_shortcut = nn.BatchNorm1d(args['emb_dims'])

        # Linear layers for regression output
        self.linear1 = nn.Linear(args['emb_dims'], 512, bias=False)
        self.bn7 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn8 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128)  # Output one scalar value
        self.bn9 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(128, 64)  # Output one scalar value
        self.bn10 = nn.BatchNorm1d(64)
        self.final_linear = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, 3, num_points).

        Returns:
            Tensor: Output tensor of the predicted scalar value.
        """
        shortcut = self.bn_shortcut(self.conv_shortcut(x))

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn6(self.conv6(x)))
        # Adding the residual connection
        x = x + shortcut

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = F.relu(self.bn7(self.linear1(x)))
        x = F.relu(self.bn8(self.linear2(x)))
        x = F.relu(self.bn9(self.linear3(x)))
        x = F.relu(self.bn10(self.linear4(x)))
        features = x
        x = self.final_linear(x)

        return x, features


class DragGNN(torch.nn.Module):
    """
    Graph Neural Network for predicting drag coefficients using GCNConv layers.

    Args:
        None

    Methods:
        forward(data): Forward pass through the network.
    """
    def __init__(self):
        super(DragGNN, self).__init__()
        self.conv1 = GCNConv(3, 512)
        self.conv2 = GCNConv(512, 1024)
        self.conv3 = GCNConv(1024, 512)
        self.fc1 = torch.nn.Linear(512, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            data (Data): Input graph data containing node features, edge indices, and batch indices.

        Returns:
            torch.Tensor: Output predictions for drag coefficients.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DragGNN_XL(torch.nn.Module):
    """
    Extended Graph Neural Network for predicting drag coefficients using GCNConv layers and BatchNorm layers.

    Args:
        None

    Methods:
        forward(data): Forward pass through the network.
    """
    def __init__(self):
        super(DragGNN_XL, self).__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 256)

        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(128)
        self.bn3 = BatchNorm(128)
        self.bn4 = BatchNorm(256)

        self.dropout = Dropout(0.4)

        self.fc = Sequential(
            Linear(256, 128),
            ReLU(),
            Dropout(0.4),
            Linear(128, 64),
            ReLU(),
            Linear(64, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            data (Data): Input graph data containing node features, edge indices, and batch indices.

        Returns:
            torch.Tensor: Output predictions for drag coefficients.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class EnhancedDragGNN(torch.nn.Module):
    """
    Enhanced Graph Neural Network for predicting drag coefficients using both GCNConv and GATConv layers,
    with Jumping Knowledge for combining features from different layers.

    Args:
        None

    Methods:
        forward(data): Forward pass through the network.
    """
    def __init__(self):
        super(EnhancedDragGNN, self).__init__()
        self.gcn1 = GCNConv(3, 64)
        self.gat1 = GATConv(64, 64, heads=4, concat=True)

        self.bn1 = BatchNorm1d(128)
        self.gcn2 = GCNConv(256, 128)
        self.gat2 = GATConv(128, 128, heads=2, concat=True)

        self.bn2 = BatchNorm1d(256)
        self.gcn3 = GCNConv(256, 256)

        self.jk = JumpingKnowledge(mode='cat')

        self.fc1 = Sequential(
            Linear(256 * 3, 128),
            ReLU(),
            BatchNorm1d(128)
        )
        self.fc2 = Linear(128, 1)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            data (Data): Input graph data containing node features, edge indices, and batch indices.

        Returns:
            torch.Tensor: Output predictions for drag coefficients.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.gcn1(x, edge_index))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.gat1(x1, edge_index)

        x2 = F.relu(self.bn1(self.gcn2(x1, edge_index)))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x2 = self.gat2(x2, edge_index)

        x3 = F.relu(self.bn2(self.gcn3(x2, edge_index)))

        x = self.jk([x1, x2, x3])

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = self.fc2(x)

        return x
