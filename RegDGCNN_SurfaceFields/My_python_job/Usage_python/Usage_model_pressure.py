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

5. forward()
    In PyTorch, the forward() method defines how your model processes input data to make a prediction
    -> example
       -> output = model(input)
       same as
       -> output = model.forward(input)
    -> forward() How data flows forward through the network

6. nn.Linear()
    -> fc = nn.Linear(Input_Channel, Out_Channel)
    -> Input shape  = [Batch_size, channel]
    -> Output_shape = [Batch_size, channel]

7. nn.Conv1d()
    -> fc = nn.Conv1d(Input_Channel, Out_Channel)
    -> Input shape  = [Batch_size, channel, Length]
    -> Output_shape = [Batch_size, channel, Length]
    -> Physical Example
       -> Channel = 3
       -> Length  = 5, value for 5 timestep
       -> channels = [
          [ 2, 4, 6, 8, 10 ],   # temperature
          [ 1, 3, 5, 7,  9 ],   # pressure
          [ 0, 1, 1, 2,  2 ]    # humidity
        ]


8. nn.Conv2d()
    -> fc = nn.Conv2d(Input_Channel, Out_Channel)
    -> Input shape  = [Batch_size, channel, Height, Width]
    -> Output_shape = [Batch_size, channel, Height, Width]

9. kernel_size
    -> Conv1d()
    -> x = [2, 4, 6, 8, 10]
    -> 1 input and 1 output
       -> kernel_size = 1, weight = 0.5, no bias
          Input : shape=(B=1, C=1, L=5)
          Output: shape=(B=1, C=1, L=5)
                  y = [2*0.5, 4*0.5, 6*0.5, 8*0.5, 10*0.5]
                    = [1.0, 2.0, 3.0, 4.0, 5.0]

       -> kernel_size = 3, weights = [0.2, 0.5, 0.3], no bias
          Apply to [0, 2, 4]:
          y[0] =  0*0.2 + 2*0.5 + 4*0.3 = 0 + 1.0 + 1.2 = 2.2

          Apply to [2, 4, 6]:
          y[1] =  2*0.2 + 4*0.5 + 6*0.3 = 0.4 + 2.0 + 1.8 = 4.2

          Apply to [4, 6, 8]:
          y[2] =  4*0.2 + 6*0.5 + 8*0.3 = 0.8 + 3.0 + 2.4 = 6.2

          Apply to [6, 8, 10]:
          y[3] =  6*0.2 + 8*0.5 + 10*0.3 = 1.2 + 4.0 + 3.0 = 8.2

          Apply to [8, 10, 0]:
          y[4] =  8*0.2 + 10*0.5 + 0*0.3 = 1.6 + 5.0 + 0 = 6.6
          -> y = [2.2, 4.2, 6.2, 8.2, 6.6]




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

    1. def __init__(self, args):
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

        #!
        self.transform = nn.Linear(256, 3*3)
        -> Create a fullt connected layer
        -> This is the final layer in "Transform_Net" that outputs a learned transformation matrix
        -> Input:  (B, 256)
        -> Output: (B, 9)  -> reshape to -> (B, 3, 3)

        #!
        init.constant_(self.transform.weight, 0)
        -> Reset all weights to 0
        -> So initially, the layer does not learn anything yet
        -> Its output only depends on the bias
        -> This ensures the transformation starts from Identity

        #!
        init.eye_(self.transform.bias.view(3, 3))
        -> self.transform.bias is a flat vector of shape (9, )
        -> view(3,3) reshape it into a matrix
        -> init.eye() fills it with an identity matrix
           [1, 0, 0]
           [0, 1, 0]
           [0, 0, 1]
    2. def forward(self, x):
       -> x is the input data
       -> x.shape = (batch_size, 128, num_points, k)

       #!
       batch_size = x.size(0)
       -> Get the size of the first dimension of the tensor x

       #!
       x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
       -> .max() function retuns two things
          -> (values, index) = x.max(dim=, keepdim=)
          -> values: the maximux value along the given dimension
          -> index : the positon of the maximum value
       -> dim=-1
          -> -1 refers to the last dimension
          -> In this case: k = neighbors
       -> [0]
          -> (values, indices)
          -> returns the max values
       -> keepdim=False
          -> Removes the reduced dimension
          -> True: keeps the reduced dimension with size 1
             -> [batch_size, 128, numpoints, 1]
        #!
        x = F.leaky_relu(self.bn4(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        -> Sequence: self.linear1(x) -> self.bn4 -> F.leadky_relu(x, slope)










