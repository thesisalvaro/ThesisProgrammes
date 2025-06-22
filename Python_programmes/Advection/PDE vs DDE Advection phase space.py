##PDE vs DDE Advection phase space, self explanatory

##Press play (takes a lot of time though)

##Comment / uncomment the final lines in order to obtain time series comparation, amplitude vs parameter comparation, etcetera

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import product

matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm
from Backup.code_backup import calculate_amplitude_period, amp_values

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



a_values = np.linspace(0.01, 0.5, 25)
q1_values = np.linspace(0.005, 0.17, 30)
q2_values = np.linspace(0.005, 0.17, 30)
k1_values = np.linspace(0.01, 0.5, 50)
k2_values = np.linspace(0.01, 0.5, 50)
K_values = np.linspace(0.01, 1.3, 30)
m1 = np.linspace(3.8, 6, 35)
m2 = np.linspace(3.8, 6, 35)
m_values = np.concatenate((m1, m2))
mu1 = np.zeros_like(m1)
mu2 = 0.03 * np.ones_like(m1)
mu_values = np.concatenate((mu1, mu2))

C = 10.0
Nx = 1501
x = np.linspace(0, C, Nx)
dx = x[1] - x[0]
delta_idx = np.abs(x-Delta).argmin()


t_span = (0, 5000)
dt = 0.005
t_steps = int(t_span[-1]/dt) 

X = np.ones([t_steps, len(m_values)])
Y = np.zeros([Nx,t_steps])


time = np.linspace(0,t_span[-1],t_steps)   
 

amplitud_finalPDE = np.zeros_like(m_values)
periodo_finalPDE = np.zeros_like(amplitud_finalPDE)

tau_valuesPDE = np.zeros_like(m_values)

##First we solve PDE

for j in tqdm(range(0, len(m_values))):
    m = m_values[j]
    mu = mu_values[j]
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

## Then we solve DDE with RungeKutta

#####


def X_equationDDE(x, y_delta, m, mu, Delta, a):
    return 1 / (1 + (np.exp(-mu*Delta/a) * y_delta)**m) - x

def Y_equationDDE(x, y, k1, k2, K, m, q1, q2):
    return ((k1 * k2 /(q1 * q2 * K ** (m+1))) * x - y) / (q1 / q2)

t_spanDDE = (0, 150)           
dtDDE = 0.0001
t_stepsDDE = int(t_spanDDE[-1]/dtDDE) 

XDDE = np.zeros([t_stepsDDE, len(m_values)])
YDDE = np.zeros([t_stepsDDE])

amplitud_finalDDE = np.zeros_like(m_values)
periodo_finalDDE = np.zeros_like(amplitud_finalDDE)

timeDDE = np.linspace(0,t_spanDDE[-1],t_stepsDDE)   


tau_values = np.zeros_like(m1)

for i in range(0, len(m1)):
    tau_values[i] = m1[i] / a

taup = Delta / a


for j in tqdm(range(0, len(m_values))):
    m = m_values[j]
    mu = mu_values[j]
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


##This is for limit cycles comparation!!!
##Uncomment / comment depending on the PDE vs DDE comparation we want


# plt.plot(XDDE_reescaled, YDDE_reescaled, label=f"DDE RungeKutta", color='b', linestyle='-.')
# plt.plot(X, Y[0, :], label=f"PDE Runge Kutta", color='r', linestyle='dashed')
# plt.plot(s1, s2, label=f'DDE ddeint', color='black')
# plt.xlabel("X(t)")
# plt.ylabel("Y(x=0,t)")
# plt.title("Phase space")
# plt.legend()
# plt.grid()


#------------------------------------------


##This is for time series comparation!!!
##Uncomment / comment depending on the PDE vs DDE comparation we want


# plt.subplot(2, 1, 1)
# # plt.plot(timeDDE_reescaled, XDDE[:,0], label=f"x(l), μ = 0", color='b', linestyle='-.')
# # plt.plot(time, X[:, 0], label=f"X(t), μ = 0", color='r', linestyle='dashed')
# # ##plt.plot(l_eval_reescaled, s1, label=f'DDE ddeint', color='black')

# # here with degradation
# # plt.plot(timeDDE_reescaled, XDDE[:,1], label=f"x(l), μ = 0.03", color='g', linestyle='-.')
# # plt.plot(time, X[:, 1], label=f"X(t), μ = 0.03", color='y', linestyle='dashed')

# # plt.xlabel("t")
# # plt.ylabel("X")
# # plt.title("Time series of X(t) vs x(l)")
# # plt.legend()
# # plt.grid()

# plt.subplot(2, 1, 2)
# plt.plot(timeDDE_reescaled, YDDE, label=f"y(l)", color='b', linestyle='-.')
# plt.plot(time, Y[0, :], label=f"Y(0, t)", color='r', linestyle='dashed')
# # plt.plot(l_eval_reescaled, s1, label=f'DDE ddeint', color='black')
# plt.xlabel("t")
# plt.ylabel("Y")
# plt.title("Time series of Y(0, t) vs y(l)")
# plt.legend()
# plt.grid()


#----------------------------------------


##This is for amplitude vs parameter comparation!!!
##Uncomment / comment depending on the PDE vs DDE comparation we want


# plt.subplot(1, 2, 1)
# plt.plot(Delta_values, amplitud_finalPDE, label=f"PDE Runge Kutta", color='r', linestyle='dashed')
# plt.plot(Delta_values, amplitud_finalDDE, label=f"DDE Runge Kutta", color='b', linestyle='-.')
# plt.xlabel("Δ")
# plt.ylabel("Amplitude")
# plt.title("Amplitude of oscillations of X(t)")
# plt.legend()
# plt.grid()


# plt.subplot(1, 2, 2)
# plt.plot(Delta_values, periodo_finalPDE, label=f"PDE Runge Kutta", color='r', linestyle='dashed')
# plt.plot(Delta_values, periodo_finalDDE, label=f"DDE RungeKutta", color='b', linestyle='-.')
# plt.xlabel("Δ")
# plt.ylabel("Period")
# plt.title("Period of oscillations of X(t)")
# plt.legend()
# plt.grid()

#------------------------------------------------


## This is amplitude/period vs delay!!!
##Uncomment / comment depending on the PDE vs DDE comparation we want


plt.subplot(2, 1, 1)
plt.plot(m_values[:len(m_values)//2], amplitud_finalPDE[:len(m_values)//2], label=f'PDE, μ = 0', color='r', linestyle='dashed')
plt.plot(m_values[:len(m_values)//2], amplitud_finalDDE[:len(m_values)//2], label=f"DDE, μ = 0", color='b', linestyle='-.')

plt.plot(m_values[len(m_values)//2:], amplitud_finalPDE[len(m_values)//2:], label=f"PDE, μ = 0.03", color='g', linestyle='dashed')
plt.plot(m_values[len(m_values)//2:], amplitud_finalDDE[len(m_values)//2:], label=f"DDE, μ = 0.03", color='y', linestyle='-.')
plt.xlabel("m")
#plt.xlabel(r'$\tau$')
plt.ylabel("Amplitude")
plt.title("Amplitude of oscillations of X(t)")
plt.legend()
plt.grid()


plt.subplot(2, 1, 2)
plt.plot(m_values[:len(m_values)//2], periodo_finalPDE[:len(m_values)//2], label=f"PDE, μ = 0", color='r', linestyle='dashed')
plt.plot(m_values[:len(m_values)//2], periodo_finalDDE[:len(m_values)//2], label=f"DDE, μ = 0", color='b', linestyle='-.')

plt.plot(m_values[len(m_values)//2:], periodo_finalPDE[len(m_values)//2:], label=f"PDE, μ = 0.03", color='g', linestyle='dashed')
plt.plot(m_values[len(m_values)//2:], periodo_finalDDE[len(m_values)//2:], label=f"DDE, μ = 0.03", color='y', linestyle='-.')
plt.xlabel("m")
#plt.xlabel(r'$\tau$')
plt.ylabel("Period")
plt.title("Period of oscillations of X(t)")
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()