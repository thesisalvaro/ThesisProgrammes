##DDE RK diffusion

##random try, not working due to bad programming of the kernel


import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm
from jitcdde import quadrature
from Backup.code_backup import amp_values, calculate_amplitude_period


def X_equation(x, y_delta, m, k1, K, q1):
    return k1 / (K**m + (y_delta)**m) - q1*x

def Y_equation(x, y, q2, k2):
    return k2*x - q2*y

def kernel(tau, D, mu, Delta):
    return (1/(np.pi*D*tau)**0.5)*np.exp(-Delta**2 /(4*D*tau) - mu*tau)


def Y1_equation(t, x, y, k2, q2, D, dx, mu): 
    y[1:-1] = (D/(dx**2))*(-2*y[1:-1]+y[:-2]+y[2:]) - mu * y[1:-1]
    y[0] = k2*x-q2*y[0]
    return y

k1 = 0.1
K = 1.0
m = 6.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
D = 0.06
mu = 0.0
Delta = 7.5
Dprime = D / q1

a_values = [0.1, 0.1125, 0.125, 0.15, 0.35, 0.3625, 0.375]
m_values = [4.8, 4.9]
q2_values = [0.03, 0.03]
k1_values = [0.08, 0.085, 0.09]
k2_values = [0.085, 0.09, 0.095, 0.1]
K_values = [0.95, 1, 1.05, 1.1, 1.15]
Delta_values = [7, 7.05, 7.1, 7.15, 7.2, 7.25]
mu_values = [0.0]

t_span = (0, 1000)           
dt = 0.01



t_steps = int(t_span[-1]/dt) 

t_spanDDE = (0, 300)           
dtDDE = 0.001
t_stepsDDE = int(t_spanDDE[-1]/dtDDE) 


XDDE = np.zeros([t_steps, len(mu_values)])
YDDE = np.zeros([t_steps])

timeDDE = np.linspace(0,t_span[-1],t_stepsDDE)

X = np.zeros([t_steps, len(mu_values)])
Y = np.zeros([t_steps, len(mu_values)])
Yprima = np.zeros_like(Y)

time = np.linspace(0,t_span[-1],t_steps)   

ker = np.zeros([t_steps, len(mu_values)])

ker3 = np.zeros_like(ker)

for j in range(0, len(mu_values)):
    mu = mu_values[j]
    for k in range(1, t_steps):
        tau = k*dt
        ker[k, j] = kernel(tau, D, mu, Delta)
    ker[:, j] /= np.sum(ker[:, j]) * dt


time = np.linspace(0,t_span[-1],t_steps)   
R = np.zeros(t_steps)

ker4 = np.sum(ker)

amplitud_final = np.zeros_like(q2_values)
periodo_final = np.zeros_like(amplitud_final)

ker2 = np.zeros_like(ker)

timeDDE_reescaled = time / q1


plt.figure(figsize=(12, 6))


for j in tqdm(range(0, len(mu_values))):
    mu = mu_values[j]
    # X[0, j] = (q1 * K**m) / k1
    X[0, j] = 1
    Y[0, j] = 0
    for i in tqdm(range(1,t_steps)):
        # integral = 0.0
        # for tau in range(i):
        #     t_k = tau * dt
        #     integral += ker[tau] * Y[i - tau - 1, j] * dt

        precomputed = ker[:i, j] * Y[i-1::-1, j]  # Reverse slice of Y up to i
        # ker2[i, j] = np.sum(precomputed)
        # precomputed /= np.sum(ker[:,j]) * dt
        cumulative = np.sum(precomputed) * dt

        # if np.sum(ker[:i, j]) == 0:
        #     ker3[i, j] = 0
        # else:
        #     ker3[i, j] = cumulative / np.sum(ker[:i,j])

        #cumulative_integral = ker3[i, j] * dt
        #cumulative = np.sum(precomputed) * dt
        cumulative_integral = cumulative
        # cumulative_integral = np.cumsum(ker[:i] * Y[i-1::-1, j]) * dt

        Yprime = cumulative
        # Yprime = cumulative_integral[-1]


        # Yprime = Y[int(i - 1 - tau/dt), j] if i - tau/dt > 0 else Y[0, j]
        k1_x = X_equation(X[i-1, j], Yprime, m, k1, K, q1)
        k1_y = Y_equation(X[i-1, j], Y[i-1, j], k2, q2)

        aux_x = X[i-1, j]+dt*k1_x
        aux_y = Y[i-1, j]+dt*k1_y
        ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
        k2_x = X_equation(aux_x, Yprime, m, k1, K, q1)
        k2_y = Y_equation(aux_x, aux_y, k2, q2)
        
        X[i, j] = X[i-1, j]+0.5*dt*(k1_x+k2_x)
        Y[i, j] = Y[i-1, j]+0.5*dt*(k1_y+k2_y)

        Yprima[i, j] = Yprime
    # amplitud_final[j], periodo_final[j] = calculate_amplitude_period(time, X[:, j])
    # XDDE[:, j] = XDDE[:, j]* k1 / (q1 * (K **m))
    # YDDE = YDDE * K



XDDE[:, j] = X[:, j]* k1 / (q1 * (K **m))


XDDE_reescaled = XDDE.astype(np.float32) * k1 /(q1 * (K **m))
YDDE_reescaled = YDDE.astype(np.float32) * K
timeDDE = time / q1


plt.plot(time, X[:, 0], label=f"X(t), μ = 0", color='b')
plt.plot(time, Yprima[:, 0], label=f"Ydelta(t), μ = 0", color='g')
plt.plot(time, Y[:,0], label=f"Y0(t), μ = 0", color='r')

plt.xlabel("t")
plt.ylabel("whateve")
plt.title("Time series of Y(0,t)")
plt.legend()
plt.grid()


plt.tight_layout()
plt.show() 