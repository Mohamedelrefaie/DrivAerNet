import numpy as np
import matplotlib.pyplot as plt

# Spatial resolution test (for alpha)
dx = np.array([0.1, 0.05, 0.025, 0.0125])
e_dx = np.array([1.5e-2, 3.8e-3, 9.5e-4, 2.4e-4])

# Time resolution test (for beta)
dt = np.array([0.01, 0.005, 0.0025, 0.00125])
e_dt = np.array([2.0e-2, 5.0e-3, 1.25e-3, 3.1e-4])

# Plotting log-log error vs dx
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.loglog(dx, e_dx, 'o-', label='Error vs dx')
z = np.polyfit(np.log(dx), np.log(e_dx), 1)
plt.loglog(dx, np.exp(z[1]) * dx ** z[0], '--', label=f"Slope ≈ {z[0]:.2f}")
plt.xlabel('Δx')
plt.ylabel('Error')
plt.title('Spatial Convergence Rate (α)')
plt.legend()
plt.grid(True, which="both", ls="--")

# Plotting log-log error vs dt
plt.subplot(1, 2, 2)
plt.loglog(dt, e_dt, 'o-', label='Error vs dt')
z = np.polyfit(np.log(dt), np.log(e_dt), 1)
plt.loglog(dt, np.exp(z[1]) * dt ** z[0], '--', label=f"Slope ≈ {z[0]:.2f}")
plt.xlabel('Δt')
plt.ylabel('Error')
plt.title('Temporal Convergence Rate (β)')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig("error_convergence.png", dpi=300)
plt.show()

