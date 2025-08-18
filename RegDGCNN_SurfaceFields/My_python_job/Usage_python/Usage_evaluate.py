"""
    Usage of evaluate.py
"""

#!---------------------------------------------------------------------------------------------
def Initialize_model(args, device):
#!
    state_dict = torch.load(args.model_checkpoint, map_location=device)
    -> Pytorch function which Loads saved object
    -> state_dict stores a model parameters i.e. weights and biases in a dictionary
    -> Just directory, not the "model architecture"
       -> This dictionary maps each layer name to its corresponding tensor
       -> example:
          {
            "conv1.weight": tensor(...),
            "conv1.bias": tensor(...),
            ...
          }

    -> args.model_checkpoint
       -> The saved model
    -> map_location=device
       -> Tell PyTorch where to load the model parameters to
       -> Especially when using different devices

#!
    model.load_state_dict(state_dict)
    -> .load_state_dict() copies the weights and biases from state_dict into your current model instance
    -> model = RegDGCNN_pressure(args_dict).to(device)
       -> This is just a Initialized model, Create the "model architecture"
       -> Which starts with random weights by default

#!---------------------------------------------------------------------------------------------
def prepare_dataset(args):
    sample_indices = list(range(len(dataset)))
    -> len(dataset)
       -> Return How many samples are in the dataset
    -> range(n)
       -> Generates a sequence of numbers from 0 to n-1
    -> list()
       -> The range object is an iterator, not a list
       -> list() converts it into an explicit Python list
       -> [0, 1, 2, 3, ...]

#!---------------------------------------------------------------------------------------------
def evaluate_model(model, dataset, sample_indices, args):
    device = next(model.parameters()).device
    -> model.parameters()
       -> Returns an iterator over all model parameter
    -> next()
       -> Get the first parameter tensor
       -> iterator is not list, it can not access by model.parameters()[0]

#!----------------
    with torch.no_grad():
    -> Do not need gradients in evaluate part
    -> Just use in train part

#!----------------
    batch_metrics = calculate_metrics(normalized_targets.cpu().numpy(), normalized_outputs.cpu().numpy())
    -> .cpu()
       -> Copy the tensor from GPU memory to CPU memory
    -> .numpy()
       -> Converts the CPU tensor into a NumPy array

#!----------------
    sample_name = os.path.basename(vtk_file).replace('.vtk', '')
    -> .os.path.basename()
       -> Get the last part of a file path
       -> i.e. the file name only
       -> example:
          -> vtk_file: /work/mae-zhangbj/ML_Turbulent/Data_Pressure_Field/Data_Pressure/PressureVTK/N_S_WWS_WM/N_S_WWS_WM_292.vtk
          -> N_S_WWS_WM_292.vtk
    -> .replace(',vtk', '')
       -> N_S_WWS_WM_292.vtk
       -> N_S_WWS_WM_292

#!----------------
    true_pressure_np = targets.cpu().numpy().squeeze()
    -> squeeze()
       -> Remove all dimensions with size 1
       -> Example:
           x1 = torch.zeros(1, 3, 1, 5)
           print(x.shape)         # (1, 3, 1, 5)
           x2 = x1.squeeze()
           print(x2.shape)        # (3, 5)

#!----------------
    for metric_name, value in all_metrics[0].items():
    -> all_metrics
       -> This is a  list stores all the validation file
    -> all_metrics[0].items()
       -> Get the first file data
       -> It returns am iterable view of key-value pairs
       -> example:
           d = {'a': 1, 'b': 2}
           print(d.items())
           ([('a', 1), ('b', 2)])

#!----------------
    agg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
    -> m[metric_name] for m in all_metrics
       -> example:
           all_metrics = [
                {'MSE': 0.01, 'MAE': 0.02, 'RMSE': 0.1},
                {'MSE': 0.02, 'MAE': 0.03, 'RMSE': 0.12},
                {'MSE': 0.015, 'MAE': 0.025, 'RMSE': 0.11}
           ]
           -> [0.01, 0.02, 0.015]

#!----------------
    np.savez(os.path.join(results_dir, 'aggregated_metrics.npz'), **agg_metrics)
    -> **agg_metrics
        -> Unpack dict keys as separate arrays
    -> Example:
        agg_metrics = {
            'MSE': np.array([0.015]),
            'MSE_std': np.array([0.004]),
            'MAE': np.array([0.02]),
            'MAE_std': np.array([0.003]),
        }
        +---------------------------------------+
        | aggregated_metrics.npz                |
        |                                       |
        |   'MSE'      --->  [0.015]            |
        |   'MSE_std'  --->  [0.004]            |
        |   'MAE'      --->  [0.020]            |
        |   'MAE_std'  --->  [0.003]            |
        +---------------------------------------+
        -> How to access after loading
        data = np.load("aggregated_metrics.npz")
        print(data['MSE'])       # [0.015]
        print(data['MAE_std'])   # [0.003]




