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

#!----------------------------------------------------------------------------
    def forward(self, x):
    x0 = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
    t = self.transform_net(x0)  # (batch_size, 3, 3)
    x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
    x = torch.bmm(x, t)    # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
    x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
    -> The above 5 lines is a module-like stuff
       -> t = self.transform_net(x0) uses edge features to predict a 3*3 transformation matrix per batch sample
       -> x = torch.bmm(x, t) applies this matrix to all points to align them before further feature extraction
          -> .bmm() is a matrix multiply function
          -> (B, N, 3) * (B, 3, 3) -> (B, 3, 3)




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

#!----------------------------------------------------------------------------
    def get_graph_feature(x, k=20, idx=None, dim9=False):

    -> x = x.view(batch_size, -1, num_points)
       -> In PyTorch, x.view() reshapes a tensor without changing its data
       -> Similar to NumPy .reshape()
       -> Must specify the new shape using dimensions
       -> -1
          -> Auto-computes this dimension size
          -> Make sure feature dimension is flexible
          -> Example:
             -> x = torch.randn(2, 3, 4)
             -> Total element: 2 × 3 × 4 = 24
             -> The -1 size: 2 × ? × 4 = 24
                             => ? = 3

#!------------------------
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    -> torch.arange(0, batch_size, device=device)
       -> Create a "1D" tensor of integers from '0' not including 'batch_size'
       -> Result: ([0, 1, 2, ..., batch_size-1])
    -> .view(-1, 1, 1)
       -> Shape changes from [batch_size-1] to (batch_size-1, 1, 1)
    -> * num_points
       -> Each batch has "num_points" points
       -> First  batch: indices [0           , num_points-1]
       -> second batch: indices [num_points-1, 2*num_points-1]
       -> ...

#!------------------------
    idx = idx + idx_base
    -> idx_base is an offset
    -> The index is changed by offset
    -> Example:
       -> idx before adjustment (batch 0): [0, 1, 2]
          idx before adjustment (batch 1): [0, 1, 2]
       -> batch 0 offset: 0
          → indices stay [0, 1, 2]
          batch 1 offset: num_points=5
          → indices become [5, 6, 7]

#!------------------------
    idx = idx.view(-1)
    -> Flatten this tensor into a "1D" vector
    -> To use these indices to directly index a flat array
    -> idx.shape = (batch_size * num_points * k)

#!------------------------
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    -> After .transpose(), the memory maybe not continuous
    -> .contiguous() make sure continuous
       -> Without it, maybe shows RuntimeError
       -> RuntimeError: view size is not compatible with input tensor's size and stride...

#!------------------------
    feature = x.view(batch_size * num_points, -1)[idx, :]
    -> Now I can find each x_j for each x_i
    -> "feature" holds the neighbor points x_j
    -> feature.shape = (batch_size * num_points * k, num_dims)
    -> x.view()
       -> Reshape x to (batch_size*num_points, num_dims)
       -> "-1" stands for the last dimension
    -> [idx, :]
       -> use "idx" to select specific points in this flattened array
          -> Specific points i.e. the "K nearest" points
          -> Each row is a point, the column is the point coordinates
          -> Select the rows from feature whose indices are given by 'idx'
       -> :
          -> Means select all columns for each row

#!------------------------
    feature = feature.view(batch_size, num_points, k, num_dims)
    -> Reshape "feature"
    -> Test
       -> logging.info(f"feature batch 0, point 0, k 0: {feature[0, 0, 0,:]}")
       -> logging.info(f"feature batch 0, point 0, k 1: {feature[0, 0, 1,:]}")
       -> logging.info(f"feature batch 0, point 2, k 0: {feature[0, 2, 0,:]}")
       -> logging.info(f"feature batch 0, point 2, k 1: {feature[0, 2, 1,:]}")
       -> point i with 'k' neareast point
       -> The last dimension is the 'k' nearest point coordinate
#!------------------------
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    -> Reshape "x" to [batch_size, num_points, k, num_dims]
    -> Repeted k times for each points

#!------------------------
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    -> feature - x
       -> means x_j - x_i
    -> torch.cat((feature - x, x), dim=3)
       -> dim = 3 means "0, 1, 2, 3" slot for shape size
       -> build a pair
          -> (x_j - x_i,  x_i)
          -> 3+3
       -> After cat
          -> num_dim(diff) + num_dim(center point) = 2 * num_dims
          -> feature.shape = (batch_size, num_points, k, 2*num_dims)
    -> .permute(0, 3, 1, 2)
       -> feature.shape = (batch_size, 2*num_dims, num_points, k)


#!----------------------------------------------------------------------------
    def knn(x, k):
    -> inner = -2 * torch.matmul(x.transpose(2, 1), x)
    -> inner.shape = (batch_size, num_points, num_points)
       -> It is an inner product operation
       -> Low dimension size Example:
          -> x.shape = (1, 2, 3)
             -> x = torch.tensor([[[1.0, 2.0, 3.0],    # x coordinates [4.0, 5.0, 6.0]]])   # y coordinates
          -> x.transpose(2, 1)
             -> [
                 [[1.0, 4.0],    # point 0
                  [2.0, 5.0],    # point 1
                  [3.0, 6.0]]    # point 2
                ]
          -> inner = torch.matmul(x.transpose(2, 1), x)
             -> inner.shape = (1, 3, 3)
             -> [
                 [P_0*P_0, P_0*P_1, P_0*P_2],
                 [P_1*P_0, P_1*P_1, P_1*P_2],
                 [P_2*P_0, P_2*P_1, P_2*P_2]
                ]
             -> Further step
                -> inner[0][0]    = [P_0*P_0, P_0*P_1, P_0*P_2]
                -> inner[0][0][0] = P_0*P_0

    -> Test code
       -> logging.info(f"inner.shape: {inner.shape}")
       -> logging.info(f"inner[0][0] value: {inner[0, 3, :]}")

#!------------------------
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    -> xx.shape = (batch_size, 1, num_points)
    -> Compute the squared norm of each point, ||x|| = x_1^2 + x_2*2 + x_3*2
    -> dim=1
       -> dim argu controls which axis you sum along
    -> keepdim=True
       -> keep the summation dimension with size 1
       -> i.e. xx.shape = (1, 1, num_points)
    -> Examples: xx.shape = [1, 1, 3]
       -> xx = [[P0_norm, P1_norm, P2_norm]]
       -> xx_T = xx.transpost(2,1)
          -> xx_T.shape = [1, 3, 1]
          -> [[P0_norm], [P1_norm], [P2_norm]]

#!------------------------
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    -> pairwise_distance is a negative value between points
    -> And I think it is bullshit about operating dimension in PyTorch!!!
       -> "I get it!" The system will automatically broadcast into same shape
       -> xx.shape            = (batch_size, 1         , num_points)
       -> inner.shape         = (batch_size, num_points, num_points)
       -> xx_T.shape          = (batch_size, num_points, 1)
       -> pair_distance.shape = (batch_size, num_points, num_points)
    -> Example:
       -> pair_distance.shape = (1 ,2, 2)
       -> [[P0-P0, P0-P1], [P1-P0], [P1-P1]]

#!------------------------
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    -> Select the top K largest values along dimension -1 i.e. The last dimension
    -> The distance is all negative, The largest value means nearest point
    -> .topk(k=k, dim=-1)
       -> Example: pairwise_distance[0, 0] = [-1, -4, -10]
          -> pairwise_distance[0, 0].topk(k=2)
          -> Return value is [-1, -4]
    -> [1]
       -> values, indices = tensor.topk(...)
       -> We just need indices
    -> Checkout
       -> value = pairwise_distance.topk(k=k, dim=-1)[0]              # (batch_size, num_points, k)
       -> idx = pairwise_distance.topk(k=k, dim=-1)[1]                # (batch_size, num_points, k)
       -> logging.info(f"point 3: {value[0,3,:], idx[0,3,:]}")


#!------------------------










