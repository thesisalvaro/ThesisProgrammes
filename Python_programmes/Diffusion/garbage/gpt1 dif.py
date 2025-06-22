##gpt a ver si haces algo bien pedazo mamaostia

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants and Parameters
C = 10.0         # Length of the spatial domain
dx = 0.1         # Spatial step
dt = 0.01       # Time step
t_span = 1000.0     # Total time for simulation
t_steps = int(t_span / dt)
x_steps = int(C / dx)


D = 0.1          # Diffusion coefficient
mu = 0.0        # Degradation rate         # Boundary condition value at x = 0
k2 = 0.08        # Source term coefficient
q2 = 0.03         # Source term degradation rate
k1 = 0.08
K = 0.9
m = 10.0
q1 = 0.03
Delta = 4.5


# Initialize Arrays
X = np.zeros([t_steps])  # Placeholder for any related variable X
Y = np.zeros([t_steps, x_steps])  # Solution array
time = np.linspace(0, t_span, t_steps)
space = np.linspace(0, C, x_steps)
delta_idx = np.abs(space-Delta).argmin()

# Initial Condition: Y = 0
Y[0, :] = 0
X[0] = 1

# Boundary Conditions
# def apply_boundary_conditions(Y_current):
#     #Y_current[0] = 0  # Dirichlet BC at x=0
#     Y_current[1] = Y_current[0]  # Neumann BC: Y_x(0, t) = 0
#     return Y_current

# Equation Definitions
def X_equation(x, y, k1, K, m, q1):
    # Example placeholder, replace with the actual equation
    return k1 / (K**m + y**m) - q1*x

def Y_equation(t, X_current, Y_current, k2, q2, D, dx, mu):
    # Diffusion-degradation equation
    Y_next = np.zeros_like(Y_current)
    Y_next[1:-1] = (D / dx**2) * (Y_current[2:] - 2 * Y_current[1:-1] + Y_current[:-2]) \
                   - mu * Y_current[1:-1]
    # Source term at the boundary (using X_current[0])
    Y_next[0] = k2 * X_current - q2 * Y_current[0]
    return Y_next

# Time Integration Loop
for i in tqdm(range(1, t_steps)):
    Y_prev = Y[i - 1, :].copy()
    # Y_prev = apply_boundary_conditions(Y_prev)  # Apply BCs at each step
    X_prev = X[i-1].copy()
    
    # First RK Step
    k1_y = Y_equation((i - 1) * dt, X[i - 1], Y_prev, k2, q2, D, dx, mu)
    k1_x = X_equation(X[i-1], Y[i-1, delta_idx], k1, K, m, q1)
    aux_y = Y_prev + dt * k1_y
    # aux_y = apply_boundary_conditions(aux_y)  # Ensure BCs are applied
    aux_x = X_prev + dt*k1_x
    
    
    # Second RK Step
    k2_y = Y_equation(i * dt, aux_x, aux_y, k2, q2, D, dx, mu)
    k2_x = X_equation(aux_x, aux_y[delta_idx], k1, K, m, q1)
    
    # Update Solution
    Y_current = Y_prev + 0.5 * dt * (k1_y + k2_y)
    # Y_current = apply_boundary_conditions(Y_current)  # Apply BCs after update
    X_current = X_prev + 0.5 * dt * (k1_x + k2_x)
    
    # Store in Solution Array
    Y[i, :] = Y_current
    X[i] = X_current

# Plot Results
plt.figure(figsize=(10, 6))
for t_idx in range(0, t_steps, int(t_steps / 10)):  # Plot at intervals
    plt.plot(space, Y[t_idx, :], label=f"t={t_idx * dt:.2f}")

# plt.plot(time, Y[:, 0])

plt.xlabel("x (space)")
plt.ylabel("Y (concentration)")
plt.title("1D Diffusion-Degradation Equation")
plt.legend()
plt.grid()
plt.show()
