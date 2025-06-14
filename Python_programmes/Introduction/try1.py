import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Parameters
k1 = 1.25   # Parameter k1
K = 2.0    # Parameter K
m = 20.0    # Exponent m
q1 = 0.03   # Parameter q1
k2 = 1.25   # Parameter k2
q2 = 0.03   # Parameter q2
a = 0.3    # Advection speed
Delta = 7.5  # Spatial parameter Delta

# Spatial grid for Y(x, t)
L = 50.0     # Length of spatial domain (approximates infinity)
nx = 500     # Number of spatial grid points
dx = L / nx  # Spatial step size
x = np.linspace(0, L, nx)

# Initial conditions for Y(x, t)
Y0_x = np.zeros(nx)  # Y(x, t=0) = 0

# Function R(t) related to X(t)
def R_prime(X):
    return 0.2 * X  # Example: R'(X(t)) = 0.2 * X(t)

# Function to calculate dX/dt and dY/dt
def system_odes(t, state, nx, dx):
    X, Y_t = state[0], state[1:]
    # Reshape Y_t back into spatial grid
    Y = Y_t.reshape((nx,))
    
    # Interpolate Y(x=Delta, t) for Y_N
    i_Delta = int(Delta / dx)
    Y_N = Y[i_Delta]
    
    # ODE for X
    dXdt = k1 / (K**m + Y_N**m) - q1 * X
    
    # Advection PDE for Y(x, t)
    dYdt = np.zeros_like(Y)
    dYdt[1:-1] = -a * (Y[2:] - Y[:-2]) / (2 * dx)  # Central difference for advection
    
    # Boundary conditions
    R_t = R_prime(X)  # R(t) = R'(X(t))
    dYdt[0] = (R_t - Y[0]) / dx  # Enforce Y(0, t) = R(t)
    dYdt[-1] = -a * (Y[-1] - Y[-2]) / dx  # Approximation for x -> infinity (Y tends to 0)
    
    # Combine into a single array
    return np.concatenate(([dXdt], dYdt))

# Initial conditions
X0 = 1.0  # Initial value for X(t)
Y_t0 = Y0_x.copy()  # Initial value for Y(x, t)

# Combine into a single state vector
state0 = np.concatenate(([X0], Y_t0))

# Time span and evaluation points
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 200)

# Solve the coupled system
solution = solve_ivp(system_odes, t_span, state0, args=(nx, dx), t_eval=t_eval)

# Extract solutions
t = solution.t
X = solution.y[0, :]
Y = solution.y[1:, :]

# Plot the results for X(t)
plt.figure(figsize=(10, 6))
plt.plot(t, X, label="X(t)")
plt.xlabel("Time (t)")
plt.ylabel("X(t)")
plt.title("Solution for X(t)")
plt.legend()
plt.grid()
plt.show()

# Plot snapshots of Y(x, t)
plt.figure(figsize=(10, 6))
for i in range(0, len(t), len(t) // 10):  # Plot 10 snapshots
    plt.plot(x, Y[:, i], label=f"t = {t[i]:.2f}")

plt.xlabel("x")
plt.ylabel("Y(x, t)")
plt.title("Snapshots of Y(x, t)")
plt.legend()
plt.grid()
plt.show()

# Heatmap of Y(x, t) against x and t
plt.figure(figsize=(10, 6))
plt.imshow(Y, extent=[t_span[0], t_span[1], x[0], x[-1]], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label="Y(x, t)")
plt.xlabel("Time (t)")
plt.ylabel("Space (x)")
plt.title("Heatmap of Y(x, t)")
plt.show()

# 3D Surface Plot of Y(x, t)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
T, X_grid = np.meshgrid(t, x)
ax.plot_surface(T, X_grid, Y, cmap='viridis', edgecolor='none')
ax.set_xlabel("Time (t)")
ax.set_ylabel("Space (x)")
ax.set_zlabel("Y(x, t)")
ax.set_title("3D Surface Plot of Y(x, t)")
plt.show()
