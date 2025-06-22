##This is a try using AI, which yielded no result

import numpy as np
import matplotlib.pyplot as plt

# Parameters
C = 10        # Spatial domain size [-C, C]
Delta = 1       # Position of coupling
a = 1         # Advection speed
k1, k2 = 10, 0.01
q1, q2 = 1, 0.01
K, m = 1, 5.0 # Constants for feedback
T = 75.0         # Total time
nx, nt = 2000, 20000  # Grid points in space, time
dx = 2 * C / nx
dt = T / nt

# Initialize arrays
x = np.linspace(-C, C, nx)
X = np.zeros(nt)  # For X(t)
Y = np.zeros((nt, nx))  # For Y(x, t)
X[0] = 1.0  # Initial condition for X
R = lambda X: 1.0 + 0.5 * X  # Boundary condition at x = 0

# Time-stepping loop
for n in range(nt-1):
    # Update X(t) (explicit method)
    coupling_term = k1 / (K**m + Y[n, int((Delta+C)/dx)]**m)
    X[n+1] = X[n] + dt * (coupling_term - q1 * X[n])
    
    # Update Y(x, t) (upwind finite difference scheme)
    for i in range(1, nx-1):
        advection = -a * (Y[n, i] - Y[n, i-1]) / dx
        Y[n+1, i] = Y[n, i] + dt * (k2 * X[n] - q2 * Y[n, i] + advection)
    
    # Boundary conditions
    Y[n+1, 0] = R(X[n+1])  # At x = 0
    Y[n+1, -1] = Y[n+1, -2]  # At x = C (absorbing)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.linspace(0, T, nt), X, label="X(t)")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.legend()

plt.subplot(1, 2, 2)
plt.imshow(Y, extent=[-C, C, 0, T], aspect='auto', origin='lower')
plt.colorbar(label="Y(x, t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Concentration Y(x, t)")

plt.tight_layout()
plt.show()
