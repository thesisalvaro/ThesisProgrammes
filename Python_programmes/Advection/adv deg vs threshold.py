##adv deg vs threshold

##PDE vs DDE Advection phase space

##Press play esta noche

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
from ddeint import ddeint
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm
from Backup.code_backup import calculate_amplitude_period, amp_values

def X_equation(x, y_delta, k1, K, m, q1):
    return k1/(np.power(K,m)+np.power(y_delta,m))-q1*x


def Y_equation(t, x, y, k2, q2, a, dx, mu): 
    y[1:] = -(a/dx)*(y[1:]-y[:-1]) - mu * y[1:]
    y[0] = k2*x-q2*y[0]
    return y

#def y0_equation(x, y0, k2, q2):
 #   return k2 * x - q2 * y0

# # Parameters
# k1 = 0.1
# K = 1
# m = 5.0
# q1 = 0.03
# q2 = 0.03
# k2 = 0.1
# a = 0.405                     # Advection speed

# Delta = 7.5                 # Position for Y(x = Delta)


# # Grid
# C = 500.0                   # Spatial domain limit
# Nx = 1001                   # Number of spatial points
# x = np.linspace(0, Delta, Nx)  # Spatial grid
# dx = x[1] - x[0]            # Spatial step
# delta_idx = np.abs(x-Delta).argmin()

k1 = 0.1
K = 1.0
m = 5.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
a = 0.35                     # Advection speed
mu = 0.0
Delta = 7.5
Kval = np.exp(0.03 * Delta/a)

a_values = np.linspace(0.01, 0.5, 25)
#m_values = [5, 5]
# q1_values = [0.1375]
q1_values = np.linspace(0.005, 0.17, 30)
q2_values = np.linspace(0.005, 0.17, 30)
k1_values = np.linspace(0.01, 0.5, 50)
k2_values = np.linspace(0.01, 0.5, 50)
K_values = [Kval]
#Delta_values = np.linspace(6, 8, 30)
#Delta_values = [7, 8]
mu_values = [0.0]




# Grid
C = 10.0                   # Spatial domain limit
Nx = 1501                   # Number of spatial points
x = np.linspace(0, C, Nx)  # Spatial grid
dx = x[1] - x[0]            # Spatial step

                 # Position for Y(x = Delta)
delta_idx = np.abs(x-Delta).argmin()

# Time span
t_span = (0, 5000)           # Time range for simulation
dt = 0.005                    # Time step for numerical evaluation
t_steps = int(t_span[-1]/dt) 

# Initial conditions
X = np.ones([t_steps, len(mu_values)])     # Initial value for X
Y = np.zeros([Nx,t_steps]) # Initial values for Y(x, t)
#Y0 = np.zeros([t_steps])
#Y0[0] = 0

time = np.linspace(0,t_span[-1],t_steps)   
 

amplitud_finalPDE = np.zeros_like(mu_values)
periodo_finalPDE = np.zeros_like(amplitud_finalPDE)

tau_valuesPDE = np.zeros_like(mu_values)

for j in tqdm(range(0, len(mu_values))):
    mu = mu_values[j]
    K = K_values[j]
    # tau = Delta / a
    # tau_valuesPDE[j] = Delta / a_values[j]
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
    amplitud_finalPDE[j], periodo_finalPDE[j] = calculate_amplitude_period(time, X[:, j])

#####

## This is DDE with RungeKutta

#####


def X_equationDDE(x, y_delta, m, mu, Delta, a):
    return 1 / (1 + (np.exp(-mu*Delta/a) * y_delta)**m) - x

def Y_equationDDE(x, y, k1, k2, K, m, q1, q2):
    return ((k1 * k2 /(q1 * q2 * K ** (m+1))) * x - y) / (q1 / q2)

t_spanDDE = (0, 150)           
dtDDE = 0.0001                    # Si cambiamos el step, hay que cambiar el round(tau, 2)
t_stepsDDE = int(t_spanDDE[-1]/dtDDE) 

##como traducir t_steps a segundos?

XDDE = np.zeros([t_stepsDDE, len(mu_values)])
# YDDE = np.zeros([t_stepsDDE, len(q2_values)])
YDDE = np.zeros([t_stepsDDE])

amplitud_finalDDE = np.zeros_like(mu_values)
periodo_finalDDE = np.zeros_like(amplitud_finalDDE)

timeDDE = np.linspace(0,t_spanDDE[-1],t_stepsDDE)   



for j in tqdm(range(0, len(mu_values))):
    mu = mu_values[j]
    K = K_values[j]
    XDDE[0, j] = (q1 * K**m) / k1
    YDDE[0] = 0
    tau = q1 * Delta / a
    tau = round(tau, 4)
    timeDDE_reescaled = timeDDE / q1
    for i in tqdm(range(1,t_stepsDDE)):
        Yprime = YDDE[int(i - 1 - tau/dtDDE)] if i - tau/dtDDE > 0 else YDDE[0]
        k1_x = X_equationDDE(XDDE[i-1, j], Yprime, m, mu, Delta, a)
        k1_y = Y_equationDDE(XDDE[i-1, j], YDDE[i-1], k1, k2, K, m, q1, q2)
        k1_y_prime = Y_equationDDE(XDDE[int(i-1 - tau/dtDDE), j], YDDE[int(i-1-tau/dtDDE)], k1, k2, K, m, q1, q2) if i - tau/dtDDE > 0 else YDDE[0]

        aux_x = XDDE[i-1, j]+dtDDE*k1_x
        aux_y = YDDE[i-1]+dtDDE*k1_y
        aux_y_prime = Yprime+dtDDE*k1_y_prime if i- tau/dtDDE > 0 else YDDE[0]
        ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
        k2_x = X_equationDDE(aux_x, aux_y_prime, m, mu, Delta, a)
        k2_y = Y_equationDDE(aux_x, aux_y, k1, k2, K, m, q1, q2)
        
        XDDE[i, j] = XDDE[i-1, j]+0.5*dtDDE*(k1_x+k2_x)
        YDDE[i] = YDDE[i-1]+0.5*dtDDE*(k1_y+k2_y)
    XDDE[:, j] = XDDE[:, j]* k1 / (q1 * (K **m))
    YDDE = YDDE * K
    amplitud_finalDDE[j], periodo_finalDDE[j] = calculate_amplitude_period(timeDDE_reescaled, XDDE[:, j])

        

XDDE_reescaled = XDDE.astype(np.float32) * k1 /(q1 * (K **m))
YDDE_reescaled = YDDE.astype(np.float32) * K
timeDDE_reescaled = timeDDE / q1



plt.figure(figsize=(12, 6))

## This is amplitude/period vs delay!!!
##This is for time series comparation!!!

plt.plot(timeDDE_reescaled, XDDE[:,0], label=r"x(l), μ = 0, K = \exp(\mu \tau)", color='b', linestyle='-.')
plt.plot(time, X[:, 0], label=f"X(t), μ = 0, K = \exp(\mu \tau)", color='r', linestyle='dashed')


###aqui con degradation
plt.plot(timeDDE_reescaled, XDDE[:,1], label=f"x(l), μ = 0.03, K = 1", color='g', linestyle='-.')
plt.plot(time, X[:, 1], label=f"X(t), μ = 0.03, K = 1", color='y', linestyle='dashed')

plt.xlabel("t")
plt.ylabel("X")
plt.title("Time series of X(t) vs x(l)")
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()