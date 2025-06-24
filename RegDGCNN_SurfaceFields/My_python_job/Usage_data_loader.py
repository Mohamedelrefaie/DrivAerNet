from torch.utils.data import Dataset, Subset, DataLoader

# ============ Class Usage ============
"""
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

4.----

************************************
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

"""

# ============ Function Usage ============
"""
    
"""
