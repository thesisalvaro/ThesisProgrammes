##Lotka-Volterra model

##Press play

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

t_span = (0, 25)
dt = 0.01
t_steps = int((t_span[-1] - t_span[0]) / dt)
time = np.linspace(t_span[0], t_span[-1], t_steps)

x = np.zeros(t_steps)
y = np.zeros(t_steps)

solx = np.zeros(t_steps)
soly = np.zeros_like(solx)

a = 1
b = 1
c = 1
d = 1

xin = [1.25, 1.5, 2, 1.5, 2, 0.5]
yin = [1, 2, 2.5, 1, 0.5, 1]

long = len(xin)
solx = np.zeros((long, t_steps))
soly = np.zeros_like(solx)

for j in range(0, long):
    x[0] = xin[j]
    y[0] = yin[j]
    for i in tqdm(range(1, t_steps)):
        k1_x = x[i-1] * (a - b * y[i-1])
        k1_y = y[i-1] * (c * x[i-1] - d)
    
        aux_x = x[i-1] + dt * k1_x
        aux_y = y[i-1] + dt * k1_y
    
        k2_x = aux_x * (a - b * aux_y)
        k2_y = aux_y * (c * aux_x - d)
    
        x[i] = x[i-1] + 0.5 * dt * (k1_x + k2_x)
        y[i] = y[i-1] + 0.5 * dt * (k1_y + k2_y)
    solx[j, :] = x
    soly[j, :] = y


plt.figure(figsize=(8, 4))

colors = ['b', 'g', 'r', 'y', 'm', 'c']


# Phase space plot (y vs. x)
x1 = np.linspace(-5, 5, 10000)
x2 = np.zeros(len(x1))

for i in range(0, long):
    plt.plot(solx[i], soly[i], label=f'(x0, y0) = ({xin[i % len(xin)]}, {yin[i % len(yin)]})', color=colors[i % len(colors)])

plt.plot(x1, x2, color='black')
plt.plot(x2, x1, color='black')
plt.xlim(-0.1, 3)
plt.ylim(-0.1, 3.1)
plt.xlabel(f'N')
plt.ylabel(f'P')
plt.title("Predator-prey model")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()