# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import product
# from tqdm import tqdm
# from scipy.signal import find_peaks
# from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.use('QtAgg')  # Use QtAgg backend for interactive plots

# --- Model equations ---
def X_equation(x, y_delta, m, k1, K, q1):
    return k1 / (K**m + y_delta**m) - q1 * x

def Y_equation(x, y, q2, k2):
    return k2 * x - q2 * y

def kernel(tau, D, mu, Delta):
    return (Delta / (4*np.pi * D * tau**3)**0.5) * np.exp(-Delta**2 / (4 * D * tau) - mu * tau)

# --- Parameters ---
k1 = 0.08
K = 0.9
m = 10.0
q1 = 0.03
q2 = 0.03
k2 = 0.08
D = 0.1
mu = 0.03
Delta = 4.5

# --- Time settings ---
t_span = (0, 5000)
dt = 0.01
t_steps = int(t_span[-1] / dt)
time = np.linspace(*t_span, t_steps)

# --- Initialize arrays ---
X = np.zeros((t_steps, 1))
Y = np.zeros((t_steps, 1))
Yprime = np.zeros_like(Y)
ker = np.zeros((t_steps, 1))
ker2 = np.zeros_like(ker)


# --- Compute kernel ---
for k in range(1, t_steps):
    tau = k * dt
    ker[k, 0] = kernel(tau, D, mu, Delta)
    # if k * dt > 10:
    #     ker[k, 0] /= np.sum(ker[:, 0])*dt
# ker[:, 0] /= np.sum(ker[:, 0]) * dt



# --- Time integration using RK2 ---
X[0, 0] = 1
Y[0, 0] = 0

for i in tqdm(range(1, t_steps)):
    precomputed = ker[:i, 0] * Y[i-1::-1, 0]

    cumulative = np.sum(precomputed) * dt

    Ydelta = cumulative

    k1_x = X_equation(X[i-1, 0], Ydelta, m, k1, K, q1)
    k1_y = Y_equation(X[i-1, 0], Y[i-1, 0], q2, k2)

    aux_x = X[i-1, 0] + dt * k1_x
    aux_y = Y[i-1, 0] + dt * k1_y

    k2_x = X_equation(aux_x, Ydelta, m, k1, K, q1)
    k2_y = Y_equation(aux_x, aux_y, q2, k2)

    X[i, 0] = X[i-1, 0] + 0.5 * dt * (k1_x + k2_x)
    Y[i, 0] = Y[i-1, 0] + 0.5 * dt * (k1_y + k2_y)

    Yprime[i, 0] = Ydelta


# --- Plot results ---
plt.figure(figsize=(12, 6))

plt.plot(time, X[:, 0], label="X(t), μ = 0")
plt.plot(time, Yprime[:, 0], label="Ydelta(t), μ = 0", linestyle='dashed')
plt.plot(time, Y[:, 0], label="Y(0,t), μ = 0")

plt.xlabel("t")
plt.ylabel("everything")
plt.title("Time series watever)")
plt.legend()
plt.grid()

###


# plt.subplot(3, 1, 1)
# plt.plot(time, X[:, 0], label="X(t), μ = 0", color='b')
# plt.xlabel("t")
# plt.ylabel("X")
# plt.title("Time series of X(t)")
# plt.legend()
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.plot(time, Yprime[:, 0], label="Ydelta(t), μ = 0", color='r', linestyle='dashed')
# plt.xlabel("t")
# plt.ylabel("Y")
# plt.title("Time series of Ydelta(t)")
# plt.legend()
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.plot(time, Y[:, 0], label="Y(0,t), μ = 0", color='y')
# plt.xlabel("t")
# plt.ylabel("Y")
# plt.title("Time series of Y(0,t)")
# plt.legend()
# plt.grid()

plt.tight_layout()
plt.show()
