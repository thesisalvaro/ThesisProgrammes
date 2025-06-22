## Advección no degradation in X and Y

#Does not really work, was one of first tries of implementing RK

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available

def X_equation(x, y_delta, k1, K, m, q1):
    return k1/(np.power(K,m)+np.power(y_delta,m))-q1*x

def X_eq(t, Xx, y_delta, k1, K, m, q1, a, dx):
    Xx[1:] = -(a/dx)*(Xx[1:]-Xx[:-1])
    Xx[0] = k1/(np.power(K,m)+np.power(y_delta,m))-q1*x
    return Xx

def Y_equation(t, x, y, k2, q2, a, dx): 
    y[1:] = -(a/dx)*(y[1:]-y[:-1])
    y[0] = k2*x-q2*y[0]
    return y

k1 = 0.1
K = 1.0
m = 5.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
a = 0.405
C = 100.0
Nx = 1001
x = np.linspace(0, C, Nx)
dx = x[1] - x[0]

Delta = 7.5
assert 0 <= Delta <= C, r"Nucleus (Delta) not positioned within the grid."


delta_idx = np.abs(x-Delta).argmin()

t_span = (0, 10000)
dt = 0.01
t_steps = int(t_span[-1]/dt) 

Xx = np.zeros([Nx, t_steps])

X = np.ones([t_steps])
Y = np.zeros([Nx,t_steps])
 
for i in tqdm(range(1,t_steps)):
    k1_x = X_eq((i-1)*dt, Xx[i-1], Y[delta_idx,i-1], k1, K, m, q1, a, dx)
    k1_y = Y_equation((i-1)*dt, Xx[i-1], Y[:,i-1].copy(), k2, q2, a, dx)
    
    aux_x = Xx[:,i-1]+dt*k1_x
    aux_y = Y[:,i-1]+dt*k1_y
    
    k2_x = X_eq(i*dt, aux_x, aux_y[delta_idx], k1, K, m, q1, a, dx)
    k2_y = Y_equation(i*dt, aux_x, aux_y, k2, q2, a, dx)
    
    Xx[:,i] = Xx[:,i-1]+0.5*dt*(k1_x+k2_x)
    Y[:,i] = Y[:,i-1]+0.5*dt*(k1_y+k2_y)
 
time = np.linspace(0,t_span[-1],t_steps)   
 
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, X, label=r"$X(t)$", color='blue')
plt.xlabel("Time")
plt.ylabel(r"$X(\Delta, t)$")
plt.title(r"Solution for $X(\Delta, t)$")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, Y[0,:], label="X(t)", color='orange')
plt.xlabel("Time")
plt.ylabel(r"$Y(0, t)$")
plt.title(r"Solution for $Y(0, t)$")
plt.legend()
plt.grid()
 
plt.tight_layout()
plt.show()