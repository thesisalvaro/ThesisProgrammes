## Diffusion DDE but solved with Runge-Kutta

##Press play

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm

def X_equation(x, y_delta, m, mu, D):
    tauprime = -2 * np.sqrt(x **2 * mu / D)
    return 1 / (1 + (np.exp(-tauprime) * y_delta)**m) - x

def Y_equation(x, y, k1, k2, K, m, q1, q2):
    return ((k1 * k2 /(K ** (m+1) * q1 * q2)) * x - y) / (q1 / q2)

k1 = 0.1
K = 0.8
m = 10.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
D = 0.2
mu = 0.01
Delta = 5


a_values = [0.1, 0.1125, 0.125, 0.15, 0.35, 0.3625, 0.375]
m_values = [4.8, 4.9]
q2_values = [0.13, 0.135, 0.14, 0.145]
k1_values = [0.08, 0.085, 0.09]
k2_values = [0.085, 0.09, 0.095, 0.1]
K_values = [0.95, 1, 1.05, 1.1, 1.15]
Delta_values = [7, 7.05, 7.1, 7.15, 7.2, 7.25]
mu_values = [0.01]

t_span = (0, 200)           
dt = 0.001
t_steps = int(t_span[-1]/dt) 

X = np.zeros([t_steps, len(mu_values)])
Y = np.zeros([t_steps, len(mu_values)])

time = np.linspace(0,t_span[-1],t_steps)   

plt.figure(figsize=(12, 6))


for j in tqdm(range(0, len(mu_values))):
    mu = mu_values[j]
    X[0, j] = 1
    Y[0, j] = 0
    tau = q1 * Delta**2 / (6*D)
    tau = round(tau, 3)
    for i in tqdm(range(1,t_steps)):
        Yprime = Y[int(i - 1 - tau/dt), j] if i - tau/dt > 0 else Y[0, j]
        k1_x = X_equation(X[i-1, j], Yprime, m, mu, D)
        k1_y = Y_equation(X[i-1, j], Y[i-1, j], k1, k2, K, m, q1, q2)
        k1_y_prime = Y_equation(X[int(i-1 - tau/dt), j], Y[int(i-1-tau/dt), j], k1, k2, K, m, q1, q2) if i - tau/dt > 0 else Y[0, j]

        aux_x = X[i-1, j]+dt*k1_x
        aux_y = Y[i-1, j]+dt*k1_y
        aux_y_prime = Yprime+dt*k1_y_prime if i- tau/dt > 0 else Y[0, j]
        ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
        k2_x = X_equation(aux_x, aux_y_prime, m, mu, D)
        k2_y = Y_equation(aux_x, aux_y, k1, k2, K, m, q1, q2)
        
        X[i, j] = X[i-1, j]+0.5*dt*(k1_x+k2_x)
        Y[i, j] = Y[i-1, j]+0.5*dt*(k1_y+k2_y)

for j in tqdm(range(0, len(mu_values))):
    plt.plot(time, X[:, j], label=f'mu ={mu_values[j]}')

plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("x(t) solution")
plt.legend()
plt.grid()
 
plt.tight_layout()
plt.show()