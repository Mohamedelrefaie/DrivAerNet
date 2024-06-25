#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu, mohamed.elrefaie@tum.de

This module is part of the research presented in the paper
"DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction".

It extends the work by providing an interactive visualization of the 3D car models in the DrivAerNet dataset.
This tutorial aims to facilitate an intuitive understanding of the dataset's structure
and the aerodynamic features of the car models it contains.

This tutorial will guide users through the process of loading, visualizing, and interacting with the 3D data
of car models from the DrivAerNet dataset. Users will learn how to navigate the dataset's file and folder structure,
visualize individual car models, and apply basic mesh operations to gain insights into the aerodynamic properties
of the models.

"""

# Data Visualization
"""

File: AeroCoefficients_DrivAerNet_FilteredCorrected.csv

This snippet demonstrates data visualization using Seaborn by creating histograms, scatter plots, 
and box plots of aerodynamic coefficients from the DrivAerNet dataset.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = '../AeroCoefficients_DrivAerNet_FilteredCorrected.csv'
data = pd.read_csv(file_path)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a figure to hold the subplots
plt.figure(figsize=(20, 10))

# Histogram of Average Cd
plt.subplot(2, 2, 1)
sns.histplot(data['Average Cd'], kde=True)
plt.title('Histogram of Average Drag Coefficient (Cd)')

# Histogram of Average Cl
plt.subplot(2, 2, 2)
sns.histplot(data['Average Cl'], kde=True)
plt.title('Histogram of Average Lift Coefficient (Cl)')

# Scatter plot of Average Cd vs. Average Cl
plt.subplot(2, 2, 3)
sns.scatterplot(x='Average Cd', y='Average Cl', data=data)
plt.title('Average Drag Coefficient (Cd) vs. Average Lift Coefficient (Cl)')

# Box plot of all aerodynamic coefficients
plt.subplot(2, 2, 4)
melted_data = data.melt(value_vars=['Average Cd', 'Average Cl', 'Average Cl_f', 'Average Cl_r'], var_name='Coefficient',
                        value_name='Value')
sns.boxplot(x='Coefficient', y='Value', data=melted_data)
plt.title('Box Plot of Aerodynamic Coefficients')

plt.tight_layout()
plt.show()


########################################################################################################################
# STL Files Visualization of the whole car
"""

Folder: DrivAerNet_STLs_Combined

This code block illustrates how to visualize 3D STL files using PyVista. The combined STL files are used for aerodynamic
drag prediction, we also provide separate STLs for the front and rear wheels (e.g. for running the CFD simulations).
Please refer to the folder: DrivAerNet_STLs_DoE
"""

import pyvista as pv
import os

# Replace with the actual path to your folder containing .stl files
folder_path = '../DrivAerNet_STLs_Combined'

# List all .stl files in the folder
stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

# Since we're going for a 2x3 grid, we'll take the first 6 .stl files for visualization
stl_files_to_visualize = stl_files[:6]

# Initialize a PyVista plotter with a 2x3 subplot grid
plotter = pv.Plotter(shape=(2, 3))

# Load and add each mesh to its respective subplot
for i, file_name in enumerate(stl_files_to_visualize):
    # Calculate the subplot position
    row = i // 3  # Integer division determines the row
    col = i % 3  # Modulus determines the column

    # Activate the subplot at the calculated position
    plotter.subplot(row, col)

    # Load the mesh from file
    mesh = pv.read(os.path.join(folder_path, file_name))

    # Add the mesh to the current subplot
    plotter.add_mesh(mesh, color='lightgrey', show_edges=True)

    # Optional: Adjust the camera position or other settings here

# Show the plotter window with all subplots
plotter.show()

########################################################################################################################
# Advanced Visualization of Mesh and Point Cloud with Pressure Data
"""

Folder: SurfacePressureVTK

This section employs PyVista to conduct an advanced visualization that includes the original 3D mesh, 
the mesh with pressure data (surface fields), and a point cloud of the mesh with pressure data. 
"""

import pyvista as pv
import numpy as np

def create_visualization_subplots(mesh, pressure_name='p', n_points=100000):
    """
    Create subplots for visualizing the solid mesh, mesh with pressure, and point cloud with pressure.

    Parameters:
    mesh (pyvista.PolyData): The mesh to visualize.
    pressure_name (str): The name of the pressure field in the mesh's point data.
    n_points (int): Number of points to sample for the point cloud.
    """
    camera_position = [(-11.073024242161921, -5.621499358347753, 5.862225824910342),
                       (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
                       (0.34000174095454166, 0.10379556639001211, 0.9346792479485448)]
    # Sample points from the mesh for the point cloud
    if mesh.n_points > n_points:
        indices = np.random.choice(mesh.n_points, n_points, replace=False)
    else:
        indices = np.arange(mesh.n_points)
    sampled_points = mesh.points[indices]
    sampled_pressures = mesh.point_data[pressure_name][indices]

    # Create a point cloud with pressure data
    point_cloud = pv.PolyData(sampled_points)
    point_cloud[pressure_name] = sampled_pressures

    # Set up the plotter
    plotter = pv.Plotter(shape=(1, 3))

    # Solid mesh visualization
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, color='lightgrey')
    plotter.add_text('Solid Mesh', position='upper_left')
    plotter.camera_position = camera_position

    # Mesh with pressure visualization
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, scalars=pressure_name, cmap='jet')
    plotter.add_scalar_bar(title=pressure_name, vertical=True)
    plotter.add_text('Mesh with Pressure', position='upper_left')
    plotter.camera_position = camera_position

    # Point cloud with pressure visualization
    plotter.subplot(0, 2)
    plotter.add_points(point_cloud, scalars=pressure_name, cmap='jet',  clim=(-600, 400), point_size=5)
    plotter.add_scalar_bar(title=pressure_name, vertical=True)
    plotter.add_text('Point Cloud with Pressure', position='upper_left')
    plotter.camera_position = camera_position
    # Show the plot
    plotter.show()

# Load your mesh data here, ensure it has the pressure data in point_data
mesh = pv.read('../SurfacePressureVTK/DrivAer_F_D_WM_WW_3000.vtk')

# Visualize the mesh, mesh with pressure, and point cloud with pressure
create_visualization_subplots(mesh, pressure_name='p', n_points=100000)


########################################################################################################################
# Visualization of Pressure and Velocity Slices in xz-plane
"""

Folder: yNormal

In this part, separate PyVista plots are created to visualize pressure ('p') and velocity ('U') slices from a VTK file. 
"""

import pyvista as pv

# Replace this with the actual path to your VTK file containing both 'p' and 'U' data
vtk_file_path = '../yNormal/DrivAer_F_D_WM_WW_3000.vtk'

# Load the VTK file
mesh = pv.read(vtk_file_path)

# Settings for a horizontal view
camera_location = (10, -30, 3)
focal_point = (10, 0, 3)
view_up = (0, 0, 1)

# Plot for 'p' (pressure)
plotter_p = pv.Plotter()  # Create a new plotter instance for 'p'
p_actor = plotter_p.add_mesh(mesh, scalars='p', cmap='jet', clim=(-600, 400), show_scalar_bar=True)
plotter_p.add_text("Pressure (p)", position='upper_left', font_size=20, color='black')
plotter_p.camera_position = [camera_location, focal_point, view_up]
plotter_p.show()  # Display the plot for 'p'

# Plot for 'U' (velocity) using the 'turbo' colormap
plotter_u = pv.Plotter()  # Create a new plotter instance for 'U'
u_actor = plotter_u.add_mesh(mesh, scalars='U', cmap='turbo', clim=(0, 30), show_scalar_bar=True)
plotter_u.add_text("Velocity (U)", position='upper_left', font_size=20, color='black')
plotter_u.camera_position = [camera_location, focal_point, view_up]
plotter_u.show()  # Display the plot for 'U'



########################################################################################################################
# Visualization of Pressure and Velocity Slices in yz-plane
"""

Folder: xNormal

In this part, separate PyVista plots are created to visualize pressure ('p') and velocity ('U') slices from a VTK file. 
"""
import pyvista as pv

# Replace this with the actual path to your VTK file containing both 'p' and 'U' data
vtk_file_path = '../xNormal/DrivAer_F_D_WM_WW_3000.vtk'

# Load the VTK file
mesh = pv.read(vtk_file_path)

# Define the final camera position for a horizontal view
final_camera_position = [(20, 0, 0),
                         (4, 0, 3),
                         (0, 0, 1)]

# Plot for 'p' (pressure)
plotter_p = pv.Plotter()  # Create a new plotter instance for 'p'
p_actor = plotter_p.add_mesh(mesh, scalars='p', cmap='jet', clim=(-600, 400), show_scalar_bar=True)
plotter_p.add_text("Pressure (p)", position='upper_left', font_size=20, color='black')
plotter_p.camera_position = final_camera_position  # Set the final camera position
plotter_p.show()  # Display the plot for 'p'

# Plot for 'U' (velocity) using the 'turbo' colormap
plotter_u = pv.Plotter()  # Create a new plotter instance for 'U'
u_actor = plotter_u.add_mesh(mesh, scalars='U', cmap='turbo', clim=(0, 30), show_scalar_bar=True)
plotter_u.add_text("Velocity (U)", position='upper_left', font_size=20, color='black')
plotter_u.camera_position = final_camera_position  # Set the final camera position
plotter_u.show()  # Display the plot for 'U'

########################################################################################################################
# Visualization of full 3D domain
"""

Folder: CFD_VTK

In this part, we visualize the full 3D domain (car and wind tunnel with boundary conditions) 
"""

import pyvista as pv

def visualize_flow_field(vtk_file_path, scalar_field='U'):
    """
    Visualize the flow field from a VTK file and allow the user to choose between 'U' (velocity) or 'p' (pressure).

    Parameters:
    vtk_file_path (str): Path to the VTK file containing the flow field data.
    scalar_field (str): Scalar field to visualize ('U' for velocity or 'p' for pressure).

    Returns:
    plotter (pyvista.Plotter): PyVista plotter object with the flow field visualization.
    """
    # Load the VTK file
    mesh = pv.read(vtk_file_path)

    # Create a plotter
    plotter = pv.Plotter()

    # Add the mesh with the selected scalar field
    if scalar_field == 'U':
        plotter.add_mesh(mesh, scalars='U', cmap='turbo', show_scalar_bar=True)
    elif scalar_field == 'p':
        plotter.add_mesh(mesh, scalars='p', cmap='jet', show_scalar_bar=True)
    else:
        raise ValueError("Invalid scalar_field value. Choose either 'U' for velocity or 'p' for pressure.")

    # Set title based on selected scalar field
    if scalar_field == 'U':
        title = "Velocity (U)"
    else:
        title = "Pressure (p)"
    plotter.add_text(title, position='upper_left', font_size=20, color='black')

    return plotter

# Example usage:
vtk_file_path = '../CFD_VTK/Exp_0003/VTK/Exp_0003_1000.vtk'
scalar_field = 'U'  # or 'p' for pressure
plotter = visualize_flow_field(vtk_file_path, scalar_field)
plotter.show()
