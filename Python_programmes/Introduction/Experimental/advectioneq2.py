import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

a = 1.0
L = 50.0
T = 5.0
nx = 500
dx = L / nx
dt = 0.01

x = np.linspace(-L/2, L/2, nx)
def initial_condition(x):
    return np.exp(-((x)**2) / 2)
Y0 = initial_condition(x)


def absorbing_layer(Y):
    damping_width = 0.1 * L
    damping_region = np.logical_or(x < -L/2 + damping_width, x > L/2 - damping_width)
    damping_factor = 0.01
    Y[damping_region] *= (1 - damping_factor)
    return Y

# Function to compute the spatial derivative using finite differences
def dydt(t, Y):
    Y = absorbing_layer(Y)  # Apply absorbing boundary conditions
    dYdx = np.zeros_like(Y)
    dYdx[1:-1] = (Y[2:] - Y[:-2]) / (2 * dx)  # Central difference
    return -a * dYdx

# Solve the system using solve_ivp
solution = solve_ivp(dydt, [0, T], Y0, method="RK45", t_eval=np.arange(0, T, dt))

# Extract results
t = solution.t
Y = solution.y

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(0, len(t), len(t) // 10):  # Plot 10 snapshots
    plt.plot(x, Y[:, i], label=f"t = {t[i]:.2f}")

plt.xlabel("x")
plt.ylabel("Y(x, t)")
plt.title("Solution of the PDE with Infinite Boundary Conditions")
plt.legend()
plt.grid()
plt.show()
