##simple limit cycle, used to investigate limit cycles

##Discarded in the end, though useful to understand limit cycles

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

t_span = (0, 25)
dt = 0.01
t_steps = int((t_span[-1] - t_span[0]) / dt)
time = np.linspace(t_span[0], t_span[-1], t_steps)

r = np.zeros(t_steps)
theta = np.zeros(t_steps)

xin = 0.01
yin = 1


mu_values = np.linspace(1, 5, 5)
rfp_valuespos = np.sqrt(mu_values)
rfp_valuesneg = -np.sqrt(mu_values)
long = len(mu_values)
solr = np.zeros((long, t_steps))
soltheta = np.zeros_like(solr)
solrdot = np.zeros_like(solr)

omega = 1
b = 20

for j in range(0, long):
    mu = mu_values[j]
    r[0] = xin
    theta[0] = yin
    for i in tqdm(range(1, t_steps)):
        k1_r = r[i-1] * (mu - r[i-1] **2)
        k1_theta = omega + b * r[i-1] **2
    
        aux_r = r[i-1] + dt * k1_r
        aux_theta = theta[i-1] + dt * k1_theta
    
        k2_r = aux_r * (mu - aux_r **2)
        k2_theta = omega + b * aux_r **2
    
        r[i] = r[i-1] + 0.5 * dt * (k1_r + k2_r)
        theta[i] = theta[i-1] + 0.5 * dt * (k1_theta + k2_theta)
    
    rdot = r * (mu - r **2)
    solr[j, :] = r
    soltheta[j, :] = theta
    solrdot[j, :] = rdot


plt.figure(figsize=(12, 6))

colors = ['b', 'b', 'b', 'b', 'b']

for i in range(0, long):
    plt.plot(solr[i], solrdot[i], label=f'μ = {mu_values[i % len(mu_values)]}', color=colors[i % len(colors)])


dot = '\u0307'

x1 = np.linspace(-5, 5, 10000)
x2 = np.zeros(len(x1))

plt.plot(x1, x2, color='black')
plt.plot(x2, x1, color='black')
plt.xlim(-.1, 2)
plt.ylim(-1.5, 5)
plt.xlabel(r'r')
plt.ylabel(r'r' + dot)
plt.title("Phase Space Plot")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()