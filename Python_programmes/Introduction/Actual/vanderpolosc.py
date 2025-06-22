##Vanderpol oscillator

## Press play

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

t_span = (0, 25)
dt = 0.01
t_steps = int((t_span[-1] - t_span[0]) / dt)
time = np.linspace(t_span[0], t_span[-1], t_steps)

x = np.zeros(t_steps)
y = np.zeros(t_steps)

xin = [2.0, -2.0, 1.0, -1.0]
yin = [0, 3.0, 0.0, 0.0]

mu_values = [0.5, 1, 1.5, 2]
long = len(xin)
solx = np.zeros((long, t_steps))
soly = np.zeros_like(solx)

for j in range(0, long):
    mu = mu_values[j]
    x[0] = xin[0]
    y[0] = yin[0]
    for i in tqdm(range(1, t_steps)):
        k1_x = y[i-1]
        k1_y = mu * (1 - x[i-1]**2) * y[i-1] - x[i-1]
    
        aux_x = x[i-1] + dt * k1_x
        aux_y = y[i-1] + dt * k1_y
    
        k2_x = aux_y
        k2_y = mu * (1 - aux_x**2) * aux_y - aux_x
    
        x[i] = x[i-1] + 0.5 * dt * (k1_x + k2_x)
        y[i] = y[i-1] + 0.5 * dt * (k1_y + k2_y)
    solx[j, :] = x
    soly[j, :] = y




plt.figure(figsize=(8, 4))

colors = ['b', 'g', 'y', 'r', 'k']


for i in range(0, long):
    plt.plot(time, solx[i], label=f'μ = {mu_values[i % len(mu_values)]}', color=colors[i % len(colors)])

plt.xlabel("Time")
plt.ylabel(r"x(t)")
plt.title(r"Van der Pol Oscillator")
plt.legend()
plt.grid()

x1 = np.linspace(-5, 5, 10000)
x2 = np.zeros(len(x1))

plt.tight_layout()
plt.show()