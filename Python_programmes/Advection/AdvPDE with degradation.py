##AdvPDE with degradation

## Press play

##This yields time series with kymograph, and can be commented /uncommented to yield time series + phase portraits or the hopf bifurcation

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm
from Backup.code_backup import amp_values, calculate_amplitude_period

def X_equation(x, y_delta, k1, K, m, q1):
    return k1/(np.power(K,m)+np.power(y_delta,m))-q1*x


def Y_equation(t, x, y, k2, q2, a, dx, mu): 
    y[1:] = -(a/dx)*(y[1:]-y[:-1]) - mu * y[1:]
    y[0] = k2*x-q2*y[0]
    return y

k1 = 0.1
K = 1.0
m = 5.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
a = 0.35
mu = 0.0
Delta = 7.5

a_values = np.linspace(0.4, 0.425, 6)
m_values = [4.8, 4.9]
q2_values = [0.015, 0.02, 0.025, 0.03, 0.13, 0.135, 0.14, 0.145]
k1_values = np.linspace(0.04, 0.5, 33)
k2_values = [0.08, 0.085, 0.09, 0.095, 0.1]
K_values = [1, 1.025, 1.05, 1.075, 1.1, 1.125]
Delta_values = [100, 500, 1000]
mu_values = [0.0]

# Grid
C = 10.0
Nx = 501
x = np.linspace(0, C, Nx)
dx = x[1] - x[0]

delta_idx = np.abs(x-Delta).argmin()

t_span = (0, 2500)
dt = 0.02
t_steps = int(t_span[-1]/dt) 

initial_values = [1]
initial_values1=[0]

X = np.ones([t_steps, len(mu_values)])
Y = np.zeros([Nx, t_steps])


time = np.linspace(0,t_span[-1],t_steps)   

amplitud_final = np.zeros_like(mu_values)
periodo_final = np.zeros_like(amplitud_final)

plt.figure(figsize=(9, 4.5))

Y1 = np.zeros([len(mu_values), t_steps])

for j in tqdm(range(0, len(mu_values))):
    X[0, j] = 1
    Y[0, 0] = 0
    mu = mu_values[j]
    for i in tqdm(range(1,t_steps)):
        k1_x = X_equation(X[i-1, j], Y[delta_idx,i-1], k1, K, m, q1)
        #k1_y0 = y0_equation(X[i-1], Y0[i-1], k2, q2)
        k1_y = Y_equation((i-1)*dt, X[i-1, j], Y[:,i-1].copy(), k2, q2, a, dx, mu)
        

        aux_x = X[i-1, j]+dt*k1_x
        #aux_y0 = Y0[i-1] + dt*k1_y0
        aux_y = Y[:,i-1]+dt*k1_y
        
        
        k2_x = X_equation(aux_x, aux_y[delta_idx], k1, K, m, q1)
        #k2_y0 = y0_equation(aux_x, aux_y0, k2, q2)
        k2_y = Y_equation(i*dt, aux_x, aux_y, k2, q2, a, dx, mu)

        
        X[i, j] = X[i-1, j]+0.5*dt*(k1_x+k2_x)
        #Y0[i] = Y0[i-1] + 0.5*dt*(k1_y0+k2_y0)
        Y[:,i] = Y[:,i-1]+0.5*dt*(k1_y+k2_y)
    #amplitud_final[j], periodo_final[j] = calculate_amplitude_period(time, X[:, j])
    Y1[j, :] = Y[0, :]
    
    #plt.plot(time, X[:, j], label="X(t)", color='blue')

 
## Uncomment here (and comment the rest) to plot Hopf bifurcation
#plt.plot(mu_values, amplitud_final, color='b', label=f'Amplitud')
#plt.plot(k1_values, periodo_final, color='r', label=f'Periodo')

# plt.xlabel("k1")
# plt.ylabel("Amplitude")
# plt.title("Hopf bifurcation")
# plt.legend()
# plt.grid()
 
##----------------

colors = ['b', 'g', 'y', 'r', 'k']



plt.subplot(2, 1, 1)

for j in tqdm(range(0, len(initial_values))):
    plt.plot(time, X[:, j])

plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("x(t) solution")
plt.legend()
plt.grid()


##Uncomment this for the phase portrait

# plt.subplot(2, 1, 2)
# for j in tqdm(range(0, len(mu_values))):
#     plt.plot(X[-100000:, j], Y1[j, -100000:], label=f'μ = {mu_values[j]}')

# plt.xlabel("X(t)")
# plt.ylabel("Y(0, t)")
# plt.title("Phase portrait")
# plt.legend()
# plt.grid()


##Uncomment here for the kymograph

Z = Y[::-1,:]
plt.subplot(2, 1, 2)
plt.imshow(Z, cmap="viridis", aspect="auto", extent=[0, t_span[-1], 0, C])
plt.colorbar()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Kymograph of Y(x, t)")

 
plt.tight_layout()
plt.show()