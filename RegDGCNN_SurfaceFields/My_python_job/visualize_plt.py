import os
import numpy as np
import matplotlib.pyplot as plt

from utils import setup_logger, setup_seed, visualize_pressure_field, plot_error_distribution, calculate_metrics


DataPath = os.path.expandvars('$HOME/ML_Turbulent/DrivAerNet/RegDGCNN_SurfaceFields/My_python_job/results/Train_Test/prediction_data/N_S_WWS_WM_292_prediction_data.npz')
data = np.load(DataPath)
points = data['points']            # shape (N_points, 3)
true_p = data['true_pressure_np']  # shape (N_points,)
pred_p = data['pred_pressure_np']  # shape (N_points,)

output_path = os.path.expandvars('$HOME/ML_Turbulent/DrivAerNet/RegDGCNN_SurfaceFields/My_python_job/results/Train_Test/visualization')
os.makedirs(output_path, exist_ok=True)

""" Use Pyvisa"""
visualize_pressure_field(points, true_p, pred_p, output_path)

""" Use matplotlib.pyplot """
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
p1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=true_p, cmap='jet', s=1)
ax1.set_title('True Pressure', fontsize=14)
fig.colorbar(p1, ax=ax1, shrink=0.5)

ax2 = fig.add_subplot(122, projection='3d')
p2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=pred_p, cmap='jet', s=1)
ax2.set_title('Predicted Pressure', fontsize=14)
fig.colorbar(p2, ax=ax2, shrink=0.5)

"""
error = np.abs(true_p - pred_p)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
p3 = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=error, cmap='hot')
ax.set_title('Prediction Error')
fig.colorbar(p3, ax=ax)
"""

plt.tight_layout()
plt.savefig(os.path.join(output_path, "matplotlib_version.png"), dpi=300)
print(f"[Info]Saved to {os.path.join(output_path, 'visualization.png')}")


