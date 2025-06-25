from torch.utils.data import Dataset, Subset, DataLoader

# ============ Class Usage ============

****class SurfacePressureDataset(Dataset):
    -> Create a custom dataset by inheriting from torch.utils.data

1.----
    def __init__(self, root_dir: str, num_points: int, preprocess=False, cache_dir=None):
    -> The constructor; runs when an instance of the class is created
#!
    self
    -> refers to the current object of the class

#!
    self.root_dir = root_dir
    -> Stores the input parameter "root_dir" inside the object

#!
    self.vtk_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.vtk')]

    os.path.join(root_dir, f)
    -> Joins the folder path "root_dir" with the filename f
    -> i.e. root_dir/f

    os.listdir(root_dir)
    -> List all files and  folder name in the "root_dir" directory

    f.endswith('.vtk')
    -> Keeps those files end with .vtk

2.----
    def __len__(self):
#!
    len(self.vtk_files)
    -> Number of .vtk files
Usage
    print(len(dataset))
    -> This will call __len__(self) automatically

3.----
    def _get_cache_path(self, vtk_file_path):
#!
    base_name = os.path.basename(vtk_file_path).replace('.vtk', '.npz')
    -> just get a string

4.----difference among the method style

    1.----
        __init__, __getitem__, __len__
    -> double underscore methods
    -> They allow your class to work with built-in python behavior

    2.----
        _get_cache_path
    -> one underscore
    -> For internal use only, but they can be accessed from outside if needed
    3.----
        sample_point_cloud_with_pressure
    -> Normal public method



# ============ Function Usage ============
1. def create_subset(dataset, ids_file):
    -> dataset:  the full SurfacePressureDataset. all .vtk files
    -> ids_file: file path
        -> "/home/heng-924/ML_Turbulent/DrivAerNet/train_val_test_splits/train_design_ids.txt"
        -> train, val, test

    #!
    try block
    -> Lets you safely run code that might crash
    -> handle errors gracefully
    -> Usage
       -> try:
            # risky code
          except SomeError:
            # handle the error

    #!
    with open(ids_file, 'r') as file:
    -> open ids_file in read mode
    -> with ... as file:
            Open this file, and automatically close

    #!
    subset_ids = [id_.strip() for id_ in file.readlines()]
     -> return a clearn list just value
        -> ['0001', '0002', '0003']
    -> file.readlines()
       -> Append '\n' for each element
       -> example
          ['0001\n', '0002\n', '0003\n']
    -> id_.strip()
       -> Remove whitespcae characters for both ends of a string

    #!
    subset_files = [f for f in dataset.vtk_files if any(id_ in f for id_ in subset_ids)]
    -> dataset.vte_files
       -> Defined in Class PressurePrediction
       -> It is a array ['001.vtk', '002.vtk', ...]
    -> any(id_ in f for id_ in subset_ids)
       -> any[True, False]  = True
       -> any[False, False] = False
       -> for id_ in subset_ids
          -> This is a loop
          -> id_ is a array stores whole subset_ids string
       -> id_ in f
          -> This is a condition
          -> id_ denotes the current string, just one string
    -> f for f in dataset.vtk_files
       -> the first f:  The value you wanna put into the new list
       -> the second f: The variable name you use to loop over the old list
    -> example
       -> dataset.vtk_files = ['car_0001.vtk', 'car_0002.vtk', 'car_0003.vtk']
          subset_ids = ['0001', '0003']
       -> subset_files = ['car_0001.vtk', 'car_0003.vtk']








2. def get_dataloaders(dataset_path: str, subset_dir: str, num_points: int, batch_size: int,
                    world_size: int, rank: int, cache_dir: str = None, num_workers: int = 4) -> tuple:

    #!
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    -> Split the dataset across multiple GPUs or processes
    -> num_replicas = world_size
       -> Total processes/GPUs will be used

    #!
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        drop_last=True, num_workers=num_workers
    )
    -> It turns a Dataset into mini-batches of data you can iterate over in a training loop.
    -> Example
       -> dataset = [data0, data1, data2, data3, ..., data9]
       -> batch_size = 4
       -> for batch in loader:
            print(batch)
       -> [ data0, data1, data2, data3 ]
          [ data4, data5, data6, data7 ]


    -> drop_last=True
       -> Drop the last incomplete batch if the dataset size is not divisible by the batech size
       -> Example
          -> Dataset has 103 samples
          -> batch_size = 12
          -> the leftover of 7 is discarded








