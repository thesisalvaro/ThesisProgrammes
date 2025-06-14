## DDE dif jitcdde
from jitcdde import jitcdde, y, t, quadrature
from sympy import symbols, sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt

# Constants
k1 = 0.1       # Reaction rate constant for X
K = 1.0        # Saturation constant for X
m = 6.0        # Hill coefficient
q1 = 0.03       # Decay rate of X
k2 = 0.1       # Reaction rate for Y
q2 = 0.03       # Decay rate of Y
D = 0.06        # Diffusion constant for the kernel
Delta = 7.5

# Define the kernel function G(tau)
def kernel(tau):
    return (1 / sqrt(4 * pi * D * tau)) * exp(-Delta**2 / (4 * D * tau))

tau = symbols("tau")

# Define the integral term: ∫ G(τ) Y(t-τ) dτ
integral_term = integral_term = quadrature(
    kernel(tau) * y(1, t - tau),  # Integrand
    tau,                         # Integration variable
    0,                           # Lower limit
    t                            # Upper limit
)

# Define the equations
X = y(0)  # State variable X(t)
Y = y(1)  # State variable Y(t)

# Differential equations
dX_dt = (k1 / (K**m + integral_term**m)) - q1 * X
dY_dt = k2 * X - q2 * Y

# Combine equations
equations = [dX_dt, dY_dt]

# Initialize the JitCDDE solver
dde = jitcdde(equations)

# Set the initial conditions
dde.constant_past([1.0, 0.0])  # Initial conditions: X(0)=1, Y(0)=0

# Start integration from t=0
dde.step_on_discontinuities()

# Define the time points to evaluate the solution
time_points = np.linspace(0, 10, 500)

# Store the results
results = []
for time in time_points:
    results.append(dde.integrate(time))

# Convert results to a numpy array for easier handling
results = np.array(results)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_points, results[:, 0], label="X(t)", color="blue")  # X(t)
plt.plot(time_points, results[:, 1], label="Y(t)", color="orange")  # Y(t)
plt.xlabel("Time (t)")
plt.ylabel("Concentration")
plt.title("Time Series of X(t) and Y(t) with Distributed Delay")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
