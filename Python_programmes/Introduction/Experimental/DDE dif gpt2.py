# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # Constants
# k1 = 0.1
# K = 1.0
# m = 6.0
# q1 = 0.03
# k2 = 0.1
# q2 = 0.03
# D = 0.06
# Delta = 7.5
# mu = 0.00

# # Time domain
# t_span = (0, 100)   # Simulation time span
# dt = 0.01           # Time step
# t_steps = int(t_span[-1] / dt)
# time = np.linspace(t_span[0], t_span[1], t_steps)

# # Initialize variables
# X = np.zeros(t_steps)
# Y = np.zeros(t_steps)
# Y_delta = np.zeros(t_steps)

# # Initial conditions
# X[0] = 1.0
# Y[0] = 0.0

# # Precompute kernel values
# tau_values = np.arange(1, t_steps) * dt  # Avoid tau=0
# kernel_values = (1 / (np.pi * D * tau_values)**0.5) * np.exp(-Delta**2 / (4 * D * tau_values) - mu * tau_values)

# # Functions
# def X_equation(x, y_delta, m, k1, K, q1):
#     return k1 / (K**m + y_delta**m) - q1 * x

# def Y_equation(x, y, q2, k2):
#     return k2 * x - q2 * y

# # Time integration
# for i in tqdm(range(1, t_steps)):
#     # Compute Y_delta (integral using kernel)
#     if i > 1:
#         integrand = kernel_values[:i] * Y[i - 1 : 0 : -1]
#         Y_delta[i] = np.sum(integrand) * dt

#     # Lax-Wendroff-like integration
#     # Predict step
#     k1_x = X_equation(X[i - 1], Y_delta[i - 1], m, k1, K, q1)
#     k1_y = Y_equation(X[i - 1], Y[i - 1], q2, k2)

#     X_mid = X[i - 1] + 0.5 * dt * k1_x
#     Y_mid = Y[i - 1] + 0.5 * dt * k1_y
#     Y_delta_mid = Y_delta[i - 1]  # Approximating as constant for small dt

#     # Correct step
#     k2_x = X_equation(X_mid, Y_delta_mid, m, k1, K, q1)
#     k2_y = Y_equation(X_mid, Y_mid, q2, k2)

#     X[i] = X[i - 1] + dt * k2_x
#     Y[i] = Y[i - 1] + dt * k2_y

# # Plot results
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 1, 1)
# plt.plot(time, X, label="X(t)")
# plt.xlabel("Time")
# plt.ylabel("X")
# plt.title("Time Series of X")
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.plot(time, Y, label="Y(t)", color='orange')
# plt.xlabel("Time")
# plt.ylabel("Y")
# plt.title("Time Series of Y")
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.plot(time, Y_delta, label="Y_delta(t)", color='green')
# plt.xlabel("Time")
# plt.ylabel("Y_delta")
# plt.title("Time Series of Y_delta")
# plt.grid()

# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 0.06  # Diffusion coefficient
tspan = 100.0  # Time
dt = 0.01
x = np.linspace(-10, 10, 500)


t_values= [0, 5, 10, 15, 20]

# Green's function and its derivative
def green_function(x, t, D):
    return 1 / np.sqrt(4* np.pi * D * t) * np.exp(-x**2 / (4 * D * t))

def green_derivative(x, t, D):
    return x / np.sqrt(4 * np.pi * D * t**3) * np.exp(-x**2 / (4 * D * t))

# Evaluate
#G = green_function(x, t, D)
#dG_dx = green_derivative(x, t, D)

# Plot
plt.figure(figsize=(10, 6))
for i in range(len(t_values)):
    plt.plot(x, green_function(x, t_values[i], D), label=f'G(x, t), t = {t_values[i]}')
    plt.plot(x, green_derivative(x, t_values[i], D), label=f'∂G/∂x, t = {t_values[i]}', linestyle="dashed")

plt.axhline(0, color="black", linewidth=0.5)
plt.xlabel("x")
plt.ylabel("Value")
plt.title("Green's Function and its Spatial Derivative")
plt.legend()
plt.grid()
plt.show()
