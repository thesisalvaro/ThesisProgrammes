import numpy as np
import matplotlib.pyplot as plt

a = 1.0
L = 10.0
T = 5.0
dx = 0.1
dt = 0.01

cfl = a * dt / dx
if cfl > 1:
    raise ValueError("CFL condition violated: reduce dt or increase dx")


x = np.arange(0, L, dx)
t = np.arange(0, T, dt)


def initial_condition(x):
    return np.exp(-((x - L/2)**2) / 2)


Y = np.zeros((len(t), len(x)))
Y[0, :] = initial_condition(x)

for n in range(0, len(t) - 1):
    for i in range(1, len(x)):
        Y[n + 1, i] = Y[n, i] - cfl * (Y[n, i] - Y[n, i - 1])
    Y[n + 1, 0] = Y[n, 0] - cfl * (Y[n, 0] - Y[n, -1])


plt.figure(figsize=(10, 6))
for i in range(0, len(t), len(t) // 10):
    plt.plot(x, Y[i, :], label=f"t = {i * dt:.2f}")
plt.xlabel("x")
plt.ylabel("Y(x, t)")
plt.title("Solution of the PDE")
plt.legend()
plt.grid()
plt.show()

