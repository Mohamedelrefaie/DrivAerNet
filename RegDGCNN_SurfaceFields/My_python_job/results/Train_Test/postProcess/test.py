import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pointNet_To_vtk import write_pointnet_vtk
import numpy as np
import pyvista as pv

pv.start_xvfb()
DataPath = os.path.expandvars('$HOME/ML_Turbulent/DrivAerNet/RegDGCNN_SurfaceFields/My_python_job/results/Train_Test/prediction_data/N_S_WWS_WM_292_prediction_data.npz')
data = np.load(DataPath)
points = data['points']            # shape (N_points, 3)
true_p = data['true_pressure_np']  # shape (N_points,)
pred_p = data['pred_pressure_np']  # shape (N_points,)

# True pressure
write_pointnet_vtk(points, true_p, fname="292_True.vtk")
write_pointnet_vtk(points, pred_p, fname="292_Pred.vtk")
