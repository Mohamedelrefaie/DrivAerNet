# model_pressure.py
"""
Model architecture for pressure field prediction on the DrivAerNet++ dataset

This module implements the RegDGCNN model for predicting pressure fields
on 3D models from the DrivAerNet++ dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np


