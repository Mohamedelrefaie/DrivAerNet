# ============ Function Usage ============
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Import modules
from model_pressure import RegDGCNN_pressure
'''

# DDP: Distributed Data Parallel
1.----
    world_size = len(gpu_list.split(','))
#! 
    gpu_list                   -> "0, 1, 2"
    gpu_list.split(',')        -> ['0', '1', '2']
    len(gpu_list.split(','))   -> 3

2.----
    exp_dir = os.path.join('experiments', args.exp_name)
#!
    Build a path "./experiments/exp_name"

3.----
    os.makedirs(exp_dir, exist_ok=True)
#!
    Create the directory "exp_dir" if it doesn't already exist

4.----
    mp.spawn(train_and_evaluate, args=(world_size, args), nprocs=world_size, join=True)
#!
   train_and_evaluate(rank, world_size, args)
   # rank :        which GPU this process is using
   # world_size :  total number of GPUS
   # args:         your parsed command-line arguments 
#!
    mp.spawn(...)
    ->
    train_evaluate(rank=0, world_size= , args=args)
    train_evaluate(rank=1, world_size= , args=args)
    train_evaluate(rank=2, world_size= , args=args)
    train_evaluate(rank=3, world_size= , args=args)
    ...
    
5.----
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
#!
    Starts the communication backend for DDP
    # nccl:        NVDIA backend for multi-GPU
    # evv://       Uses evvironmen variables(e.g. RANK, WORLD_SIZE, etc)

6.-----
    torch.cuda.set_device(local_rank)
#! 
    Each process uses a single GPU
    Set the GPU this process will use

7.-----
    args = vars(args)
#!
    Convert it to a regular dictionary
    e.g. {'epoch': 10, 'lr': 0.0001}

8.-----
    model = RegDGCNN_pressure(args).to(local_rank)
#!
    RegDGCNN_pressure(args): Creates an instance of your custom model class
    .to(local_rank):         Moves the model to the current corret GPU
#!
    The detail please see Usage_model_pressure.pu

10.----
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True, 
        output_device=local_rank
    )
#!
    Wrap it with DDP so Pytorch handles multi-GPU
        synchronization and data parallelism
#!
    model
    The model aleady moved to the local GPU
#!
    device_ids=[local_rank]
    Restrict this process to use only one GPU
#!
    find_unused_parameters=True
    It tells pyTorch "Some layers in my model might not be used every time I call forward()
    -> so please handle that correctly"
        The detail please see Usage_model_pressure.py forwar() function

#!  
    output_device=local_rank
    Ensures outputs go to the same GPU as inputs
    
11.----
    criterion = torch.nn.MSELoss()
#!
    Loss function:
    This sets up Mean Squared Error(MSE) as the loss function, commonly used for regression problems
    It measures the average of the squares of the differences between predicted and actual values:
        MSE = 1/n * sum((yi - y)^2)      i = 1, ... , n
    
12.----
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#!
    Uses Adam(Adaptive Moment Estimation), a popular optimizer that adjusts learning rates for each parameter
#!
    model.parameters()    
    ->Tell the optimizer which parameters to update
#!
    lr = args.lr
    ->learning rate 
#!
    weight_decay=1e-4
    ->adds L2 regularization to reduce overfitting

13.----
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)
#!    
    Automatically reduces the learning rate when the validation loss plateaus(stops improving)
#!
    'min'
    ->Try to minimize val loss
    -> By default 1e-4

#!
    patience=10 
    ->Wait for 10 epochs without improvement

#!
    factor=0.1
    ->Multiplies LR by 0.1

#!
    e.g.
    Epoch 1 - Validation Loss: 0.470
    LR : 0.01
    Epoch 2 - Validation Loss: 0.471
    LR : 0.01
    Epoch 3 - Validation Loss: 0.471
    LR : 0.01
    Epoch 4 - Validation Loss: 0.470
    LR : 0.01*0.1
    
14.----
    train_dataloader.sampler.set_epoch(epoch)
    -> DDP needed function
    -> Helps ensure different GPU processes don't get the same data every epoch

15.----
    train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank)
    -> function for training
    
#!
    model.train()
    -> bulit-in function for nn.Module API
    -> set the model to training mode

#!
    for data, targets in tqdm(train_dataloader, desc="[Training]"):
    -> tqdm is just a process bar
    -> [Training]:  56%|█████████████████▌         | 50/90 [00:05<00:04,  8.23it/s]
    -> Tells you are on batch 50/90
    -> Is the same as " for data, targets in train_dataloader "

#!
    data, targets = data.squeeze(1).to(local_rank), targets.squeeze(1).to(local_rank)
    -> .to(local_rank) sent data to GPU "local_rank"
    -> .squeeze(1) rm the element "1" in the data and target
    -> I do not know "1" stands for what

#!
    targets = (targets - PRESSURE_MEAN) / PRESSURE_STD
    -> Normalizes the ground truth targets(pressure values)
PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25

#!
    optimizer.zero_grad()
    -> Clears previous gradients stored in the model(from last batch)

#!
    outputs = model(data)
    -> outputs could be predicted pressure values
    -> Triggers DDP's _call_() method
    -> DDP calls forward(data) function
    -> forward() defined in model_pressure.py
    -> Equivalent to outputs = model.forward(data)

#!
    loss = criterion(outputs.squeeze(1), targets)
    -> MSE function for Pressure part
    -> loss = ((outputs - targets)**2).mean()

#!
    loss.backward()
    -> Computes gradients of the loss w.r.t all model parameters
    -> See model_pressure.py backward() function

#!
    optimizer.step()
    -> Updates the model's weights using the computed gradients in loss.backward()

#!
    total_loss += loss.item()
    -> loss.item() converts the scalar tensor to a python number

#!
    return total_loss / len(train_dataloader)
    -> Returns the average loss per batch over the entire epoch
    -> len(train_dataloader) the number of batches passed from the command-line
    
'''














