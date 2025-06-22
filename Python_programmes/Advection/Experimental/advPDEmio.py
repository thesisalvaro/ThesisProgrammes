##This is the exact same as the advPDE_final_method, except it does not work for some reason

##It goes awry for some parameters, did not delve too deep in why due to time concerns as always

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm

def X_equation(x, y_delta, k1, K, m, q1):
    return k1/(np.power(K,m)+np.power(y_delta,m))-q1*x


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
a = 0.35
Delta = 7.5

C = 10.0
Nx = 1001
x = np.linspace(0, C, Nx)
dx = x[1] - x[0]

delta_idx = np.abs(x-Delta).argmin()

t_span = (0, 5000)
dt = 0.05
t_steps = int(t_span[-1]/dt)

X = np.ones([t_steps])
Y = np.zeros([Nx,t_steps])


for i in tqdm(range(1,t_steps)):
    k1_x = X_equation(X[i-1], Y[delta_idx,i-1], k1, K, m, q1)
    #k1_y0 = y0_equation(X[i-1], Y0[i-1], k2, q2)
    k1_y = Y_equation((i-1)*dt, X[i-1], Y[:,i-1].copy(), k2, q2, a, dx)
    

    aux_x = X[i-1]+dt*k1_x
    #aux_y0 = Y0[i-1] + dt*k1_y0
    aux_y = Y[:,i-1]+dt*k1_y
    
    
    k2_x = X_equation(aux_x, aux_y[delta_idx], k1, K, m, q1)
    #k2_y0 = y0_equation(aux_x, aux_y0, k2, q2)
    k2_y = Y_equation(i*dt, aux_x, aux_y, k2, q2, a, dx)
    
    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
    #Y0[i] = Y0[i-1] + 0.5*dt*(k1_y0+k2_y0)
    Y[:,i] = Y[:,i-1]+0.5*dt*(k1_y+k2_y)


 
time = np.linspace(0,t_span[-1],t_steps)   

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, X, label="X(t)", color='blue')
plt.xlabel("Time")
plt.title("Solution for X(t)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.imshow(Y[::-1,:], cmap="viridis", aspect="auto", extent=[0,t_span[-1],0,Nx])
plt.colorbar()
plt.xlabel("Time")
plt.ylabel("Y(x, t)")
plt.title("Solution for Y(x, t)")

 
plt.tight_layout()
plt.show()