import os
import numpy as np
import matplotlib.pyplot as plt

from utils import setup_logger, setup_seed, visualize_pressure_field, plot_error_distribution, calculate_metrics

DataPath = os.path.expandvars('$HOME/ML_Turbulent/DrivAerNet/RegDGCNN_SurfaceFields/My_python_job/results/Train_Test/prediction_data/N_S_WWS_WM_292_prediction_data.npz')
data = np.load(DataPath)
points = data['points']            # shape (N_points, 3)
true_p = data['true_pressure_np']  # shape (N_points,)
pred_p = data['pred_pressure_np']  # shape (N_points,)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121, projection='3d')
p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=true_p, cmap='jet')
ax.set_title('True Pressure')
fig.colorbar(p, ax=ax)

ax2 = fig.add_subplot(122, projection='3d')
p2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=pred_p, cmap='jet')
ax2.set_title('Predicted Pressure')
fig.colorbar(p2, ax=ax2)

"""
error = np.abs(true_p - pred_p)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
p3 = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=error, cmap='hot')
ax.set_title('Prediction Error')
fig.colorbar(p3, ax=ax)
"""

visualization_path = os.path.expandvars('$HOME/ML_Turbulent/DrivAerNet/RegDGCNN_SurfaceFields/My_python_job/results/Train_Test/visualization')
os.makedirs(visualization_path, exist_ok=True)

plt.savefig(os.path.join(visualization_path, "visualization.png"), dpi=300)
print(f"Saved to {os.path.join(visualization_path, 'visualization.png')}")


