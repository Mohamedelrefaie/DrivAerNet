import numpy as np
import matplotlib.pyplot as plt

data = np.load('./prediction_data/N_S_WWS_WM_292_prediction_data.npz')
points = data['points']            # shape (N_points, 3)
true_p = data['true_pressure_np']  # shape (N_points,)
pred_p = data['pred_pressure_np']  # shape (N_points,)

fig = plt.figure(figsize=(10, 5))
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

plt.savefig("visualization.png", dpi=300)
print("Saved to visualization.png")


