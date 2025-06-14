## adv DDE but solved with Runge-Kutta
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm
from Backup.code_backup import amp_values, calculate_amplitude_period


def X_equation(x, y_delta, m, mu, Delta, a):
    return 1 / (1 + (np.exp(-mu*Delta/a) * y_delta)**m) - x

def Y_equation(x, y, k1, k2, K, m, q1, q2):
    return ((k1 * k2 /(K ** (m+1) * q1 * q2)) * x - y) / (q1 / q2)

k1 = 0.1
K = 1.0
m = 5.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
a = 0.35                     # Advection speed
mu = 0.0
Delta = 7.5
D = 0.03



a_values = [0.1, 0.1125, 0.125, 0.15, 0.35, 0.3625, 0.375]
m_values = [4.8, 4.9]
q2_values = [0.03]
k1_values = [0.08, 0.085, 0.09]
k2_values = [0.085, 0.09, 0.095, 0.1]
K_values = [0.95, 1, 1.05, 1.1, 1.15]
Delta_values = [7, 7.05, 7.1, 7.15, 7.2, 7.25]
mu_values = [0.0]

t_span = (0, 2000)           
dt = 0.001                    # Si cambiamos el step, hay que cambiar el round(tau, 2)
t_steps = int(t_span[-1]/dt) 

##como traducir t_steps a segundos?

X = np.zeros([t_steps, len(q2_values)])
Y = np.zeros([t_steps, len(q2_values)])

time = np.linspace(0,t_span[-1],t_steps)   

amplitud_final = np.zeros_like(q2_values)
periodo_final = np.zeros_like(amplitud_final)

plt.figure(figsize=(12, 6))


for j in tqdm(range(0, len(q2_values))):
    q2 = q2_values[j]
    X[0, j] = 1
    Y[0, j] = 0
    # tau = q1 * Delta / a
    tau = Delta**2 / (6*D)
    tau = round(tau, 3)
    for i in tqdm(range(1,t_steps)):
        Yprime = Y[int(i - 1 - tau/dt), j] if i - tau/dt > 0 else Y[0, j]
        k1_x = X_equation(X[i-1, j], Yprime, m, mu, Delta, a)
        k1_y = Y_equation(X[i-1, j], Y[i-1, j], k1, k2, K, m, q1, q2)
        k1_y_prime = Y_equation(X[int(i-1 - tau/dt), j], Y[int(i-1-tau/dt), j], k1, k2, K, m, q1, q2) if i - tau/dt > 0 else Y[0, j]

        aux_x = X[i-1, j]+dt*k1_x
        aux_y = Y[i-1, j]+dt*k1_y
        aux_y_prime = Yprime+dt*k1_y_prime if i- tau/dt > 0 else Y[0, j]
        ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
        k2_x = X_equation(aux_x, aux_y_prime, m, mu, Delta, a)
        k2_y = Y_equation(aux_x, aux_y, k1, k2, K, m, q1, q2)
        
        X[i, j] = X[i-1, j]+0.5*dt*(k1_x+k2_x)
        Y[i, j] = Y[i-1, j]+0.5*dt*(k1_y+k2_y)
    # amplitud_final[j], periodo_final[j] = calculate_amplitude_period(time, X[:, j])

# for j in tqdm(range(0, len(mu_values))):
#     plt.plot(time, X[:, j], label=f'mu ={mu_values[j]}')

# plt.xlabel("l")
# plt.ylabel("x(l)")
# plt.title("x(l) solution")
# plt.legend()
# plt.grid()


# for j in tqdm(range(0, len(q2_values))):
#     amplitud_final[j] = calculate_amplitude_period(time, X[:, j])


plt.plot(time, X[:, 0])
#plt.plot(q2_values, amplitud_final)

plt.xlabel("q2")
plt.ylabel("Amplitude")
plt.title("hopf bifurcation")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()