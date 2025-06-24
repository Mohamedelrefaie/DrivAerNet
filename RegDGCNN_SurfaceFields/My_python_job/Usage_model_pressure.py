import torch
import torch.nn as nn

# ====== Machinery Learning Knowledhe ======
1. hyperparameter
    -> A hyperparameter is configuration value that you set before training a machine learning model
    -> It is not learned from data
    -> It controls how the model is trained or structured
    e.g.
    learning rate, batch_size, droupout

2. Batch Normalization
    -> BatchNorm helps deep learning models train faster and more stable
    -> Normalizing the outputs of a layer (i.e. make the data mean=0, std=1)
    -> Then leaning how to scale and shift them again

    #!
    -> This helps reduce the risk of "vanishing gradients"
    -> And makes lreaning less sensitive to initialization or learning rate

3. Channel
    -> A channel is a set of values at each point that descirbe different types of information
    -> Think of it as a feature layer stacked on top of another

    #! Real-World Analogy
    1. Imagine a color image:
      -> A color image has 3 channel:
      -> Red, Green, and Blue
    2. At each pixel (x,y), have 3 values:
      -> Red value, Green value, Blue value
    3. So an image os shape (3, H, W) has
      -> 3 channels(RGC)
      -> Height H (Number of pixel top to bottom)
      -> Width  W (Number of pixel left to right)
    e.g. (3, 224, 224)

    #! Current case 3D Point Cloud
    Maybe
    start with 6 channels x, y, z corrdinates
      and p_i, p_j, p_k
      the relative neighbor
    -> Input:   shape = (B, 6, N)
    -> Conv2d:  shape = (B, 64, N, K)
    -> B:  batch size
    -> N:  number of points
    -> K:  neighbors
    -> 64: number of channels(feature types learned by model)

4. Fully Connected Layer

    self.linear1 = nn.Linear(1024, 512, bias=False)
    self.bn4 = nn.BatchNorm1d(512)
    -> This type layer connects every input neuron to every output neuron using a weight
    -> The weight is PyTorch defined and User can modify
    -> example
    Input:   [x1, x2, x3]

    Output:  y1 = w11*x1 + w12*x2 + w13*x3 + b1
             y2 = w21*x1 + w22*x2 + w23*x3 + b2

# ====== Class Knowledge ======
1.
class RegDGCNN_SurfaceFields(nn.Module):
    -> This defines a custom Pytorch Neural Network
    -> It inherits from nn.Moudle, which is the base class for all PyTorch models

    #!
    super().__init__()
    -> Calls the parent class nn.Module constructor to set up internal PyTorch machinery

2.
class Transform_Net(nn.Module):
    -> This modules learns a transformation matrix to align the input point cloud or local features

    #!
    self.k = 3
    -> Final transformation output is a 3*3 matrix

    #!
    self.bn1 = nn.BatchNorm2d(64)
    -> 64 means: the input has 64 channels
    -> It is used after a "Conv2d" layer that output shape maybe (batch_size, 64, height, width)

    #!
    nn.LeakyReLU(negative_slope=0.2)
    -> It is an activation function in PyTorch
    -> Introduces non-linearity into your model
    -> Allows neural networks to learn complex patterns e.g. curves, edges, relationships
    -> example, alpha = 0.2
      -> f(x) = x       if x > 0
              = α * x   if x ≤ 0
      -> x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
         act = nn.LeakyReLU(negative_slope=0.2)
         y = act(x)
         print(y)  # tensor([-0.6000, -0.2000,  0.0000,  1.0000,  3.0000])

    #!
    nn.Conv2d(6, 64, kernel_size=1, bias=False)
    -> Creates a 2D convolutional layer in PyTorch
    -> It processes input data and learns to extract meaningful features from it
    -> 6:             Input  channel
    -> 64:            Output channel
    -> kernel_size=1: Use a 1*1 filter i.e. pointwise convolution
    -> bias=False:    Don not add a learnable bias. Often used with BatchNorm after

    #!
    self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                               self.bn1,
                               nn.LeakyReLU(negative_slope=0.2))
    -> Define a layer: self.conv1
    -> nn.Sequential() executes the layer in order
    -> This "conv1" layer executes this Sequence: Input -> Conv2d -> BatchNorm2d -> LeakyReLU -> Output

    #!
    self.linear1 = nn.Linear(1024, 512, bias=False)
    self.bn4 = nn.BatchNorm1d(512)
    -> Fully Connected Layer
    -> Use BatchNorm make data more correct









