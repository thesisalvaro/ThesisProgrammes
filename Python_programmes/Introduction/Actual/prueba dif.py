# ##prueba dif

##This is some random diffusion simulation, press play

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.integrate as integrate
from tqdm import tqdm
matplotlib.use('QtAgg')


def X_equation(x, y_delta, m, k1, K, q1):
    return k1 / (K**m + y_delta**m) - q1 * x

z = np.zeros(499)

def Y_equation(t, y, D, dx, mu): 

    y[1:-1] = (D/(dx**2))*(-2*y[1:-1]+y[:-2]+y[2:] - mu * y[1:-1])
    y[0] = 1
    y[-1] = 0
    return y

def kernel(tau, D, mu, Delta):
    return (1 / (np.pi * D * tau)**0.5) * np.exp(-Delta**2 / (4 * D * tau) - mu * tau)

def integrand(t, x, D, mu):
    return (x) /((4*np.pi * D *(t)**3)**0.5) * np.exp(-x**2 / (4 * D * (t)) - mu * (t))

def ker2(t, x, D, mu):
    return integrate.quad(integrand, 0, t, args=(x, D, mu, t))

span = [0, 5]
sx = 101
space = np.linspace(span[0], span[1], sx)
ddx = space[1] - space[0]

k1 = 0.1
K = 1.0
m = 6.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
D = 0.06
mu = 0.0
Delta = 7.5

t_span = (0, 100)
dt = 0.01
t_steps = int(t_span[-1] / dt)
time = np.linspace(*t_span, t_steps)

C = 5.0
Nx = 501
x = np.linspace(0, C, Nx)
dx = x[1] - x[0]      

delta_idx = np.abs(x-Delta).argmin()

cfl_limit = (dx**2) / (2 * D)
if dt > cfl_limit:
    print(f"(dt = {dt}) > CFL limit ({cfl_limit:.5f}). Set dt to {cfl_limit:.5f}.")
    dt = cfl_limit

Y = np.zeros((Nx, t_steps))
sol = np.zeros([sx, t_steps])
Y[0, 0] = 1

for i in tqdm(range(1, t_steps)):

    Y[:, 0] = 0
    Y[0, 0] = 1

    k1_y = Y_equation(i, Y[:, i-1], D, dx, mu)

    aux_y = Y[:, i-1] + dt * k1_y

    k2_y = Y_equation(i, aux_y, D, dx, mu)

    Y[:, i] = Y[:, i-1] + 0.5 * dt * (k1_y + k2_y)

plt.figure(figsize=(8, 4))

for t_idx in range(0, t_steps, t_steps // 5):
    plt.plot(x, kernel(time[t_idx], D, mu, x), label=f't = {time[t_idx]:.2f}', linestyle='dashed')

plt.xlabel("x")
plt.ylabel("Y(x,t)")
plt.title("Diffusion equation for a single pulse")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()