#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:54:56 2023

@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper":
"DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction".

The module defines a PyTorch Dataset for loading and transforming 3D car models from the DrivAerNet dataset
stored as STL files.
It includes functionality to subsample or pad the vertices of the models to a fixed number of points as well as
visualization methods for the DrivAerNet dataset.
"""
import os
import logging
import torch
import numpy as np
import pandas as pd
import trimesh
from torch.utils.data import Dataset
import pyvista as pv
import seaborn as sns
from typing import Callable, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataAugmentation:
    """
    Class encapsulating various data augmentation techniques for point clouds.
    """
    @staticmethod
    def translate_pointcloud(pointcloud: torch.Tensor, translation_range: Tuple[float, float] = (2./3., 3./2.)) -> torch.Tensor:
        """
        Translates the pointcloud by a random factor within a given range.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            translation_range: A tuple specifying the range for translation factors.

        Returns:
            Translated point cloud as a torch.Tensor.
        """
        # Randomly choose translation factors and apply them to the pointcloud
        xyz1 = np.random.uniform(low=translation_range[0], high=translation_range[1], size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return torch.tensor(translated_pointcloud, dtype=torch.float32)

    @staticmethod
    def jitter_pointcloud(pointcloud: torch.Tensor, sigma: float = 0.01, clip: float = 0.02) -> torch.Tensor:
        """
        Adds Gaussian noise to the pointcloud.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute value for noise.

        Returns:
            Jittered point cloud as a torch.Tensor.
        """
        # Add Gaussian noise and clip to the specified range
        N, C = pointcloud.shape
        jittered_pointcloud = pointcloud + torch.clamp(sigma * torch.randn(N, C), -clip, clip)
        return jittered_pointcloud

    @staticmethod
    def drop_points(pointcloud: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
        """
        Randomly removes points from the point cloud based on the drop rate.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            drop_rate: The percentage of points to be randomly dropped.

        Returns:
            The point cloud with points dropped as a torch.Tensor.
        """
        # Calculate the number of points to drop
        num_drop = int(drop_rate * pointcloud.size(0))
        # Generate random indices for points to drop
        drop_indices = np.random.choice(pointcloud.size(0), num_drop, replace=False)
        # Drop the points
        keep_indices = np.setdiff1d(np.arange(pointcloud.size(0)), drop_indices)
        dropped_pointcloud = pointcloud[keep_indices, :]
        return dropped_pointcloud

class DrivAerNetDataset(Dataset):
    """
    PyTorch Dataset class for the DrivAerNet dataset, handling loading, transforming, and augmenting 3D car models.
    """
    def __init__(self, root_dir: str, csv_file: str, num_points: int, transform: Optional[Callable] = None):

        """
        Initializes the DrivAerNetDataset instance.

        Args:
            root_dir: Directory containing the STL files for 3D car models.
            csv_file: Path to the CSV file with metadata for the models.
            num_points: Fixed number of points to sample from each 3D model.
            transform: Optional transform function to apply to each sample.
        """
        self.root_dir = root_dir
        # Attempt to load the metadata CSV file and log errors if unsuccessful
        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f"Failed to load CSV file: {csv_file}. Error: {e}")
            raise

        self.transform = transform  # Transformation function to be applied to each sample
        self.num_points = num_points  # Number of points each sample should have
        self.augmentation = DataAugmentation()  # Instantiate the DataAugmentation class

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def min_max_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data to the range [0, 1] based on min and max values.

        Args:
            data: Input data as a torch.Tensor.

        Returns:
            Normalized data as a torch.Tensor.
        """
        # Compute min and max values for normalization
        # Assuming data is a PyTorch Tensor of shape [num_points, num_features]
        min_vals, _ = data.min(dim=0, keepdim=True)  # Keep dimension for broadcasting
        max_vals, _ = data.max(dim=0, keepdim=True)  # Keep dimension for broadcasting

        # Ensure min_vals and max_vals are Tensors compatible for broadcasting
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    def _sample_or_pad_vertices(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Subsamples or pads the vertices of the model to a fixed number of points.

        Args:
            vertices: The vertices of the 3D model as a torch.Tensor.
            num_points: The desired number of points for the model.

        Returns:
            The vertices standardized to the specified number of points.
        """
        num_vertices = vertices.size(0)
        # Subsample the vertices if there are more than the desired number
        if num_vertices > num_points:
            indices = np.random.choice(num_vertices, num_points, replace=False)
            vertices = vertices[indices]
        # Pad with zeros if there are fewer vertices than desired
        elif num_vertices < num_points:
            padding = torch.zeros((num_points - num_vertices, 3), dtype=torch.float32)
            vertices = torch.cat((vertices, padding), dim=0)
        return vertices
    def __getitem__(self, idx: int, apply_augmentations: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample and its corresponding label from the dataset, with an option to apply augmentations.

        Args:
            idx (int): Index of the sample to retrieve.
            apply_augmentations (bool, optional): Whether to apply data augmentations. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sample (point cloud) and its label (Cd value).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract the relevant row from the DataFrame using the provided index
        row = self.data_frame.iloc[idx]
        design_id = row['Design']
        cd_value = row['Average Cd']
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")

        # Load the STL file and handle errors
        try:
            mesh = trimesh.load(geometry_path, force='mesh')
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise

        # Convert mesh vertices to a tensor and standardize to the specified number of points
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        vertices = self._sample_or_pad_vertices(vertices, self.num_points)

        # Apply data augmentations if enabled
        if apply_augmentations:
            vertices = self.augmentation.translate_pointcloud(vertices.numpy())
            vertices = self.augmentation.jitter_pointcloud(vertices)

        # Apply optional transformations
        if self.transform:
            vertices = self.transform(vertices)

        # Normalize the features of the point cloud
        point_cloud_normalized = self.min_max_normalize(vertices)

        cd_value = torch.tensor(float(cd_value), dtype=torch.float32).view(-1)
        #return point_cloud_normalized, cd_value
        return vertices, cd_value

    # Visualization methods for the DrivAerNetDataset class

    def visualize_mesh(self, idx):
        """
        Visualize the STL mesh for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function loads the mesh from the STL file corresponding to the design ID at the given index,
        wraps it using PyVista for visualization, and then sets up a PyVista plotter to display the mesh.
        """
        # Retrieve design ID and construct the file path for the STL file
        row = self.data_frame.iloc[idx]
        design_id = row['Design']
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")

        # Attempt to load the mesh from the STL file and handle potential errors
        try:
            mesh = trimesh.load(geometry_path, force='mesh')
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise

        # Convert the trimesh mesh to a PyVista mesh for visualization
        pv_mesh = pv.wrap(mesh)

        # Set up the PyVista plotter
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color='lightgrey', show_edges=True)
        plotter.add_axes()

        # Define a specific camera position for a consistent viewing angle
        camera_position = [(-11.073024242161921, -5.621499358347753, 5.862225824910342),
                           (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
                           (0.34000174095454166, 0.10379556639001211, 0.9346792479485448)]
        plotter.camera_position = camera_position

        # Display the plotter window with the mesh
        plotter.show()

    def visualize_mesh_withNode(self, idx):
        """
        Visualizes the mesh for a specific design from the dataset with nodes highlighted.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function loads the mesh from the STL file and highlights the nodes (vertices) of the mesh using spheres.
        It uses seaborn to obtain visually distinct colors for the mesh and nodes.
        """
        # Retrieve design ID and construct the file path for the STL file
        row = self.data_frame.iloc[idx]
        design_id = row['Design']
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")

        # Attempt to load the mesh from the STL file and handle potential errors
        try:
            mesh = trimesh.load(geometry_path, force='mesh')
            pv_mesh = pv.wrap(mesh)
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise

        # Set up the PyVista plotter
        plotter = pv.Plotter()
        sns_blue = sns.color_palette("colorblind")[0]  # Using seaborn to get a visually distinct blue color

        # Add the mesh to the plotter with light grey color and black edges
        plotter.add_mesh(pv_mesh, color='lightgrey', show_edges=True, edge_color='black')

        # Highlight nodes (vertices) of the mesh as blue spheres
        nodes = pv_mesh.points
        plotter.add_points(nodes, color=sns_blue, point_size=10, render_points_as_spheres=True)

        # Add axes for orientation and display the plotter window
        plotter.add_axes()
        plotter.show()

    def visualize_point_cloud(self, idx):
        """
        Visualizes the point cloud for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function retrieves the vertices for the specified design, converts them into a point cloud,
        and uses the z-coordinate for color mapping. PyVista's Eye-Dome Lighting is enabled for improved depth perception.
        """
        # Retrieve vertices and corresponding CD value for the specified index
        vertices, _ = self.__getitem__(idx)
        vertices = vertices.numpy()

        # Convert vertices to a PyVista PolyData object for visualization
        point_cloud = pv.PolyData(vertices)
        colors = vertices[:, 2]  # Using the z-coordinate for color mapping
        point_cloud["colors"] = colors  # Add the colors to the point cloud

        # Set up the PyVista plotter
        plotter = pv.Plotter()

        # Add the point cloud to the plotter with color mapping based on the z-coordinate
        plotter.add_points(point_cloud, scalars="colors", cmap="Blues", point_size=3, render_points_as_spheres=True)

        # Enable Eye-Dome Lighting for better depth perception
        plotter.enable_eye_dome_lighting()

        # Add axes for orientation and display the plotter window
        plotter.add_axes()
        camera_position = [(-11.073024242161921, -5.621499358347753, 5.862225824910342),
                           (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
                           (0.34000174095454166, 0.10379556639001211, 0.9346792479485448)]

        # Set the camera position
        plotter.camera_position = camera_position

        plotter.show()
    def visualize_augmentations(self, idx):
        """
        Visualizes various augmentations applied to the point cloud of a specific design in the dataset.

        Args:
            idx (int): Index of the sample in the dataset to be visualized.

        This function retrieves the original point cloud for the specified design and then applies a series of augmentations,
        including translation, jittering, and point dropping. Each version of the point cloud (original and augmented) is then
        visualized in a 2x2 grid using PyVista to illustrate the effects of these augmentations.
        """
        # Retrieve the original point cloud without applying any augmentations
        vertices, _ = self.__getitem__(idx, apply_augmentations=False)
        original_pc = pv.PolyData(vertices.numpy())

        # Apply translation augmentation to the original point cloud
        translated_pc = self.augmentation.translate_pointcloud(vertices.numpy())
        # Apply jitter augmentation to the translated point cloud
        jittered_pc = self.augmentation.jitter_pointcloud(translated_pc)
        # Apply point dropping augmentation to the jittered point cloud
        dropped_pc = self.augmentation.drop_points(jittered_pc)

        # Initialize a PyVista plotter with a 2x2 grid for displaying the point clouds
        plotter = pv.Plotter(shape=(2, 2))

        # Display the original point cloud in the top left corner of the grid
        plotter.subplot(0, 0)  # Select the first subplot
        plotter.add_text("Original Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(original_pc, color='black', point_size=3)  # Add the original point cloud to the plot

        # Display the translated point cloud in the top right corner of the grid
        plotter.subplot(0, 1)  # Select the second subplot
        plotter.add_text("Translated Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(pv.PolyData(translated_pc.numpy()), color='lightblue', point_size=3)  # Add the translated point cloud to the plot

        # Display the jittered point cloud in the bottom left corner of the grid
        plotter.subplot(1, 0)  # Select the third subplot
        plotter.add_text("Jittered Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(pv.PolyData(jittered_pc.numpy()), color='lightgreen', point_size=3)  # Add the jittered point cloud to the plot

        # Display the dropped point cloud in the bottom right corner of the grid
        plotter.subplot(1, 1)  # Select the fourth subplot
        plotter.add_text("Dropped Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(pv.PolyData(dropped_pc.numpy()), color='salmon', point_size=3)  # Add the dropped point cloud to the plot

        # Display the plot with all point clouds
        plotter.show()


# Example usage
#if __name__ == '__main__':
    #dataset = DrivAerNetDataset(root_dir='../DrivAerNet_STLs_Combined',
    #                                 csv_file='../AeroCoefficients_DrivAerNet_FilteredCorrected.csv',
    #                                 num_points=500000)

    #dataset.visualize_mesh_withNode(300)  # Visualize the mesh of the first sample

    #dataset.visualize_point_cloud(300)  # Visualize the point cloud of the first sample
