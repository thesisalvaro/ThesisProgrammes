import numpy as np
import matplotlib.pyplot as plt

# Parameters
k1 = 1.25   # Parameter k1
K = 2.0    # Parameter K
m = 1.0    # Exponent m
q1 = 0.03   # Parameter q1
k2 = 1.25   # Parameter k2
q2 = 0.03   # Parameter q2
a = 0.4017    # Advection speed
Delta = 7.5  # Spatial parameter Delta

L = 50.0    # Length of spatial domain
T = 100.0    # Total time
Nx = 500    # Number of spatial grid points
Nt = 1000   # Number of time steps
dx = L / (Nx - 1)  # Spatial step size
dt = T / Nt        # Time step size

# Spatial grid
x = np.linspace(0, L, Nx)

# Initial conditions
X = 1.0  # X(0) = 1
Y = np.zeros(Nx)  # Y(x, 0) = 0

# Arrays to store solutions
X_history = np.zeros(Nt)  # To store X(t) over time
Y_history = np.zeros((Nt, Nx))  # To store Y(x, t) over time

# Function for R(t) = R'(X(t))
def R(X):
    return X  # Example: R(t) = X(t)

# Time-stepping loop
for n in range(Nt):
    # Store current values
    X_history[n] = X
    Y_history[n, :] = Y

    # Update X(t) using forward Euler
    Y_N = Y[int(Delta / dx)]  # Y_N = Y(x = Delta, t)
    dX_dt = k1 / (K**m + Y_N**m) - q1 * X
    X += dX_dt * dt

    # Update Y(x, t) using forward Euler for the ODE part
    dY_dt_ODE = k2 * X - q2 * Y
    Y += dY_dt_ODE * dt

    # Update Y(x, t) using finite difference for the advection part
    dY_dx = np.zeros(Nx)
    dY_dx[1:-1] = (Y[2:] - Y[:-2]) / (2 * dx)  # Central difference
    dY_dx[0] = (Y[1] - Y[0]) / dx  # Forward difference at x = 0
    dY_dx[-1] = (Y[-1] - Y[-2]) / dx  # Backward difference at x = L
    Y += -a * dY_dx * dt

    # Apply boundary conditions
    Y[0] = R(X)  # Y(0, t) = R(t) = R'(X(t))
    Y[-1] = 0    # Y(L, t) = 0 (approximation for lim x -> infinity)

# Plot results
plt.figure(figsize=(12, 6))

# Plot X(t)
plt.subplot(1, 2, 1)
plt.plot(np.linspace(0, T, Nt), X_history, label="X(t)")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("X(t) over time")
plt.legend()

# Plot Y(x, t) at final time
plt.subplot(1, 2, 2)
plt.plot(x, Y_history[-1, :], label=f"Y(x, t={T})")
plt.xlabel("x")
plt.ylabel("Y(x, t)")
plt.title(f"Y(x, t) at t = {T}")
plt.legend()

plt.tight_layout()
plt.show()