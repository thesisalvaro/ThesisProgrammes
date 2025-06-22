import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
k1 = 10.0
K = 1.0
m = 2.0
q1 = 0.5
q2 = 0.2
k2 = 0.3
a = 1.0  # Advection speed
C = 100.0  # Spatial domain limit
Delta = 1.0  # Position for Y(x = Delta)
Nx = 10000  # Number of spatial points
x = np.linspace(-C, C, Nx)  # Spatial grid
dx = x[1] - x[0]  # Spatial step

# Initial conditions
X0 = 1.0  # Initial value for X
Y0 = np.zeros(Nx)  # Initial values for Y(x, t)

# Time span
t_span = (0, 100)  # Time range for simulation
dt = 0.01  # Time step for numerical evaluation

# Helper function to find the index for x = Delta
def find_nearest(array, value):
    return (np.abs(array - value)).argmin()

delta_idx = find_nearest(x, Delta)

# Define the initial condition function f(t - x/a)
def initial_condition(t_minus_x_over_a):
    """Function representing the initial condition Y(x, t) = f(t - x/a)."""
    return np.maximum(0, -np.sin(np.pi * t_minus_x_over_a))  # Example function


# Define the system of equations
def system_rhs(t, U):
    X = U[0]
    Y = U[1:]

    # Compute dY/dt for all spatial points
    dY_dt = np.zeros_like(Y)
    for i in range(1, Nx - 1):
        # Compute advection using the function f(t - x/a)
        t_minus_x_over_a = t - x[i] / a
        advection = initial_condition(t_minus_x_over_a)
        reaction = k2 * X - q2 * Y[i]
        dY_dt[i] = reaction - advection

    # Boundary conditions
    dY_dt[0] = 0  # Y(x = -C, t) = 0
    dY_dt[-1] = 0  # Y(x = C, t) = 0

    # Compute dX/dt
    Y_delta = Y[delta_idx]
    dX_dt = k1 / (K**m + Y_delta**m) - q1 * X

    return np.concatenate(([dX_dt], dY_dt))

# Combine initial conditions
U0 = np.concatenate(([X0], Y0))

# Time evaluation points
t_eval = np.arange(t_span[0], t_span[1], dt)

# Solve the system using solve_ivp
sol = solve_ivp(system_rhs, t_span, U0, t_eval=t_eval, method='RK45')

# Extract solutions
X_sol = sol.y[0]
Y_sol = sol.y[1:]

# Plot the results
plt.figure(figsize=(12, 6))

# Plot X(t)
plt.subplot(2, 1, 1)
plt.plot(sol.t, X_sol, label="X(t)", color='blue')
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("Solution for X(t)")
plt.legend()
plt.grid()

# Plot Y(x, t) for selected x values
plt.subplot(2, 1, 2)
for i in range(0, Nx, Nx // 2):
    plt.plot(sol.t, Y_sol[i], label=f"Y(x={x[i]:.1f}, t)")
plt.xlabel("Time")
plt.ylabel("Y(x, t)")
plt.title("Evolution of Y(x, t) at different x points")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
