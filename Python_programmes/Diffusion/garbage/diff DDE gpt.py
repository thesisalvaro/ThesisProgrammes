## gpt pde dif

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
k1 = 0.1
K = 1.0
m = 6.0
q1 = 0.03
k2 = 0.1
q2 = 0.03
D = 0.06
Delta = 7.5

t_span = (0, 10000)  # Time span
dt = 0.005         # Time step
t_steps = int(t_span[-1] / dt)

# # # # # Precompute kernel G(tau)
# # # # tau_values = np.arange(1, t_steps) * dt  # Avoid tau = 0
# # # # G_values = 1 / np.sqrt(np.pi * D * tau_values) * np.exp(-Delta**2 / (4 * D * tau_values))

# Regularization parameter for tau (to avoid kernel singularity)
tau_min = 400  # Cutoff time for regularizing kernel

# Precompute kernel G(tau) with regularization
tau_values = np.arange(1, t_steps) * dt  # Avoid tau = 0
G_values = np.zeros(t_steps - 1)
for k, tau in enumerate(tau_values):
    if tau >= tau_min:
        G_values[k] = 1 / np.sqrt(np.pi * D * tau) * np.exp(-Delta**2 / (4 * D * tau))

# Renormalize the kernel
G_values /= np.sum(G_values) * dt

# Initialize arrays
X = np.zeros(t_steps)
Y = np.zeros(t_steps)
integral_G_Y = np.zeros(t_steps)

# Initial conditions
X[0] = 1.0
Y[0] = 0.0

# Functions
def X_equation(X_current, Y_integral, m, k1, K, q1):
    return k1 / (K**m + Y_integral**m) - q1 * X_current

def Y_equation(X_current, Y_current, k2, q2, D):
    return k2 * X_current - q2 * Y_current / np.sqrt(np.pi *D)

# Time loop
for i in tqdm(range(1, t_steps)):
    # Compute running integral of G(tau) * Y(t - tau)
    if i > 1:
        integral_G_Y[i] = integral_G_Y[i-1] + G_values[i-2] * Y[i-1] * dt
    
    # First RK step
    k1_x = X_equation(X[i-1], integral_G_Y[i], m, k1, K, q1)
    k1_y = Y_equation(X[i-1], Y[i-1], k2, q2, D)
    
    aux_x = X[i-1] + dt * k1_x
    aux_y = Y[i-1] + dt * k1_y
    
    # Second RK step
    k2_x = X_equation(aux_x, integral_G_Y[i], m, k1, K, q1)
    k2_y = Y_equation(aux_x, aux_y, k2, q2, D)
    
    # Update X and Y
    X[i] = X[i-1] + 0.5 * dt * (k1_x + k2_x)
    Y[i] = Y[i-1] + 0.5 * dt * (k1_y + k2_y)

# Plot results
time = np.linspace(t_span[0], t_span[1], t_steps)
plt.figure(figsize=(12, 6))
plt.plot(time, X, label="X(t)")
plt.plot(time, Y, label="Y(t)")
plt.plot(time, integral_G_Y, label="Y_delta")
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Solutions X(t) and Y(t)")
plt.legend()
plt.grid()
plt.show()
