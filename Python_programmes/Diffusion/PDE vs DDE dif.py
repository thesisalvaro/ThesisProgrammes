##PDE vs DDE diffusion

##Press play, comment / uncomment to get different time series + kymograph, time series + phase portraits, etc

##This takes a lot, and I mean a lot of time. The convoluting integral takes a lot of time when times are bigger than 1k

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm
from Backup.code_backup import calculate_amplitude_period, amp_values

def X_equationPDE(x, y_delta, k1, K, m, q1):
    return k1/(np.power(K,m)+np.power(y_delta,m))-q1*x


def Y_equationPDE(t, x, y, k2, q2, D, dx, mu): 
    y[1:-1] = (D/(dx**2))*(-2*y[1:-1]+y[:-2]+y[2:]) - mu * y[1:-1]
    y[0] = k2*x-q2*y[0]
    return y


## CHOOSE PARAMETERS FOR BOTH -------------------------------


k1 = 0.08
K = 0.9
m = 10.0
q1 = 0.03
q2 = 0.03
k2 = 0.08
D = 0.1
mu = 0.03
Delta = 4.5

##PARAMETERS TO VARY ----------------------------------------

a_values = np.linspace(0.4, 0.425, 6)
m_values = np.linspace(8, 15, 140)
m_values2 = np.linspace(8, 15, 140)
q2_values = [0.015, 0.02, 0.025, 0.03, 0.13, 0.135, 0.14, 0.145]
k1_values = [0.085, 0.09, 0.095, 0.1]
k2_values = [0.08, 0.085, 0.09, 0.095, 0.1]
K_values = [0.8]
Delta_values = np.linspace(2, 9, 140)
Delta_values2 = np.linspace(0, 11.2, 140)
#Delta_values = [4, 4.5, 5]
mu_values = [0.0, 0.03]
D_values = [0.06]

#------------------------------------------------------------

C = 100.0
Nx = 1001
x = np.linspace(0, C, Nx)
dx = x[1] - x[0]

t_spanPDE = (0, 15000)
dtPDE = 0.01     

tauPDE = np.zeros_like(Delta_values)
taudeg = np.zeros_like(tauPDE)

for i in range(0, len(Delta_values)):
    tauPDE[i] = (Delta_values[i])**2 /(6*D)
    taudeg[i] = (1/(4*mu)) *(-3 + np.sqrt(9 + (4*mu*Delta_values2[i]**2 )/(D))) 


cfl_limit = (dx**2) / (2 * D)
if dtPDE > cfl_limit:
    print(f"(dt = {dtPDE}) > CFL limit ({cfl_limit:.5f}). Set dt to {cfl_limit:.5f}.")
    dtPDE = cfl_limit

t_stepsPDE = int(t_spanPDE[-1]/dtPDE) 


downsample_factor = 10  # Factor to downsample time steps
downsampled_steps = t_stepsPDE // downsample_factor
downsampled_steps = (t_stepsPDE + downsample_factor - 1) // downsample_factor

Y_full = None  # This will hold the full data during computation

Y_prev = np.zeros(Nx)
Y_current = np.zeros(Nx)

X_downsampled = np.zeros((downsampled_steps, len(m_values)))


time_downsampled = np.linspace(0, t_spanPDE[-1],downsampled_steps)

amplitud_finalPDE = np.zeros((len(m_values), len(mu_values)))
periodo_finalPDE = np.zeros_like(amplitud_finalPDE)

XPDE = np.ones([t_stepsPDE, len(m_values)])


time = np.linspace(0,t_spanPDE[-1],t_stepsPDE)   

##This is the PDE resolution

for k in tqdm(range(0, len(mu_values))):
    mu = mu_values[k]
    for j in tqdm(range(0, len(m_values))):
        if k == 0:
            m = m_values[j]
        else:
            m = m_values2[j]
        delta_idx = np.abs(x-Delta).argmin()

        for i in tqdm(range(1,t_stepsPDE)):
            k1_x = X_equationPDE(XPDE[i-1, j], Y_prev[delta_idx], k1, K, m, q1)
            k1_y = Y_equationPDE((i-1)*dtPDE, XPDE[i-1, j], Y_prev.copy(), k2, q2, D, dx, mu)
        
            aux_x = XPDE[i-1, j]+dtPDE*k1_x
            aux_y = Y_prev+dtPDE*k1_y
        
            k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
            k2_y = Y_equationPDE(i*dtPDE, aux_x, aux_y, k2, q2, D, dx, mu)
        
            XPDE[i, j] = XPDE[i-1, j]+0.5*dtPDE*(k1_x+k2_x)
            Y_current = Y_prev+0.5*dtPDE*(k1_y+k2_y)

        # Downsample data
            if i % downsample_factor == 0:
                downsample_index = i // downsample_factor
                if downsample_index < downsampled_steps:
                    X_downsampled[downsample_index, j] = XPDE[i, j]
                #Y_downsampled[:, downsample_index, j] = Y_current

        # Update previous state
            Y_prev = Y_current.copy()

        amplitud_finalPDE[j, k], periodo_finalPDE[j, k] = calculate_amplitude_period(time, XPDE[:, j])


##SOLS:
    ## X_downsampled[downsampled tsteps, len(mu)]]
    ## Y_downsampled[Nx, downsampled tsteps, len(mu)]]


#-----------------
#END OF THE PDE
#-----------------
#START OF THE DDE
#-----------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.use('QtAgg')  # Use QtAgg backend for interactive plots


def X_equationDDE(x, y_delta, m, k1, K, q1):
    return k1 / (K**m + y_delta**m) - q1 * x

def Y_equationDDE(x, y, q2, k2):
    return k2 * x - q2 * y

def kernel(tau, D, mu, Delta):
    return (Delta / (4*np.pi * D * tau**3)**0.5) * np.exp(-Delta**2 / (4 * D * tau) - mu * tau)

t_spanDDE = (0, 5000)
dtDDE = 0.01
t_stepsDDE = int(t_spanDDE[-1] / dtDDE)
timeDDE = np.linspace(*t_spanDDE, t_stepsDDE)


XDDE = np.zeros((t_stepsDDE, len(mu_values)))
YDDE= np.zeros((t_stepsDDE, len(mu_values)))
Yprime = np.zeros_like(YDDE)
ker = np.zeros((t_stepsDDE, 1))

amplitud_finalDDE = np.zeros_like(mu_values)
periodo_finalDDE = np.zeros_like(amplitud_finalDDE)


for k in range(1, t_stepsDDE):
    tau = k * dtDDE
    ker[k, 0] = kernel(tau, D, mu, Delta)

XDDE[0, 0] = 1
YDDE[0, 0] = 0

for j in tqdm(range(0, len(mu_values))):
    mu = mu_values[j]
    for i in tqdm(range(1, t_stepsDDE)):
        precomputed = ker[:i, 0] * YDDE[i-1::-1, 0]

        cumulative = np.sum(precomputed) * dtDDE

        Ydelta = cumulative

        k1_x = X_equationDDE(XDDE[i-1, 0], Ydelta, m, k1, K, q1)
        k1_y = Y_equationDDE(XDDE[i-1, 0], YDDE[i-1, 0], q2, k2)

        aux_x = XDDE[i-1, 0] + dtDDE * k1_x
        aux_y = YDDE[i-1, 0] + dtDDE * k1_y

        k2_x = X_equationDDE(aux_x, Ydelta, m, k1, K, q1)
        k2_y = Y_equationDDE(aux_x, aux_y, q2, k2)

        XDDE[i, 0] = XDDE[i-1, 0] + 0.5 * dtDDE * (k1_x + k2_x)
        YDDE[i, 0] = YDDE[i-1, 0] + 0.5 * dtDDE * (k1_y + k2_y)

        Yprime[i, 0] = Ydelta

    amplitud_finalDDE[j], periodo_finalDDE[j] = calculate_amplitude_period(timeDDE, XDDE[:, j])



##SOLS:
    ## XDDE[t_stepsDDE, len(mu)]]
    ## YDDE[t_stepsDDE, len(mu)]]



#--------------------
#THIS IS THE GRAPHING PART
#--------------------



plt.figure(figsize=(12,6))

##This is for limit cycles comparation!!!
##Uncomment this / comment the others

# plt.plot(XDDE, YDDE, label=f"DDE RungeKutta", color='b', linestyle='-.')
# plt.plot(X_downsampled, Y_downsampled[0, :], label=f"PDE Runge Kutta", color='r', linestyle='dashed')
# plt.xlabel("X(t)")
# plt.ylabel("Y(x=0,t)")
# plt.title("Phase space")
# plt.legend()
# plt.grid()


#----------------------------------


##This is for time series comparation!!
##Uncomment this / comment the others


# plt.subplot(2, 1, 1)
# plt.plot(timeDDE, XDDE[:, 0], label=f'DDE', color='b')
# plt.plot(time_downsampled, X_downsampled[:, 0], label=f'PDE', color='r', linestyle='dashed')

# ##aqui con degradation
# # plt.plot(timeDDE, XDDE[:, 1], label=f"DDE, μ = 0.03", color='g', linestyle='-.')
# # plt.plot(time_downsampled, X_downsampled[:, 1], label=f"PDE, μ = 0.03", color='y', linestyle='dashed')

# plt.xlabel("t")
# plt.ylabel("X")
# plt.title("Time series of X(t) for PDE and DDE simulations")
# plt.legend()
# plt.grid()

# plt.subplot(2, 1, 2)
# plt.plot(timeDDE, YDDE[:, 0], label=f"DDE", color='b')
# plt.plot(time_downsampled, Y_downsampled[0, :, 0], label=f"PDE", color='r', linestyle='dashed')

# # plt.plot(timeDDE, YDDE[:, 1], label=f"DDE, μ = 0.03", color='g', linestyle='-.')
# # plt.plot(time_downsampled, Y_downsampled[0, :, 1], label=f"PDE, μ = 0.03", color='y', linestyle='dashed')

# plt.xlabel("t")
# plt.ylabel("Y")
# plt.title("Time series of Y(0, t) for PDE and DDE simulations")
# plt.legend()
# plt.grid()


#------------------------------


##This is for amplitude vs parameter comparation!!!
##Uncoment this / comment the others

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

## Paupsita
# print('m_values', m_values)
# print('tau_values normales', tau_values)
#print('tau_values sin la q_1, aka DDE reescaled', tau_valuesDDE_reescaled)


#---------------------------------------


## This plots amplitude/period vs delay!!!
## Uncomment this / comment the others

# plt.subplot(2, 1, 1)
# plt.plot(tauPDE, amplitud_finalPDE[:, 0], label=f'PDE, μ = 0', color='r', linestyle='dashed')
# plt.plot(tauPDE, amplitud_finalDDE[:, 0], label=f"DDE, μ = 0", color='b', linestyle='-.')

# plt.plot(taudeg, amplitud_finalPDE[:, 1], label=f"PDE, μ = 0.03", color='g', linestyle='dashed')
# plt.plot(taudeg, amplitud_finalDDE[:, 1], label=f"DDE, μ = 0.03", color='y', linestyle='-.')

# plt.xlabel(r'$\tau_{max}$')
# plt.ylabel("Amplitude")
# plt.title("Amplitude of oscillations of X(t)")
# plt.legend()
# plt.grid()
# #μ = 0

# plt.subplot(2, 1, 2)
# plt.plot(tauPDE, periodo_finalPDE[:, 0], label=f"PDE, μ = 0", color='r', linestyle='dashed')
# plt.plot(tauPDE, periodo_finalDDE[:, 0], label=f"DDE, μ = 0", color='b', linestyle='-.')

# plt.plot(taudeg, periodo_finalPDE[:, 1], label=f"PDE, μ = 0.03", color='g', linestyle='dashed')
# plt.plot(taudeg, periodo_finalDDE[:, 1], label=f"DDE, μ = 0.03", color='y', linestyle='-.')


# plt.xlabel(r'$\tau_{max}$')
# plt.ylabel("Period")
# plt.title("Period of oscillations of X(t)")
# plt.legend()
# plt.grid()


#------------------------------------------


##This plots amplitude/period vs m!


plt.subplot(2, 1, 1)
plt.plot(m_values, amplitud_finalPDE[:, 0], label=f'PDE, μ = 0.', color='r', linestyle='dashed')
plt.plot(m_values, amplitud_finalDDE[:, 0], label=f"DDE. μ = 0", color='b', linestyle='-.')

plt.plot(m_values, amplitud_finalPDE[:, 1], label=f"PDE, μ = 0.03", color='g', linestyle='dashed')
plt.plot(m_values, amplitud_finalDDE[:, 1], label=f"DDE, μ = 0.03", color='y', linestyle='-.')
plt.xlabel("m")
#plt.xlabel(r'$\tau$')
plt.ylabel("Amplitude")
plt.title("Amplitude of oscillations of X(t)")
plt.legend()
plt.grid()


plt.subplot(2, 1, 2)
plt.plot(m_values, periodo_finalPDE[:, 0], label=f"PDE, μ = 0", color='r', linestyle='dashed')
plt.plot(m_values, periodo_finalDDE[:, 0], label=f"DDE, μ = 0", color='b', linestyle='-.')

plt.plot(m_values, periodo_finalPDE[:, 1], label=f"PDE, μ = 0.03", color='g', linestyle='dashed')
plt.plot(m_values, periodo_finalDDE[:, 1], label=f"DDE, μ = 0.03", color='y', linestyle='-.')

# plt.plot(m_values[len(m_values)//2:], periodo_finalPDE[len(m_values)//2:], label=f"PDE, μ = 0.03", color='g', linestyle='dashed')
# plt.plot(m_values[len(m_values)//2:], periodo_finalDDE[len(m_values)//2:], label=f"DDE, μ = 0.03", color='y', linestyle='-.')
plt.xlabel("m")
#plt.xlabel(r'$\tau$')
plt.ylabel("Period")
plt.title("Period of oscillations of X(t)")
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()