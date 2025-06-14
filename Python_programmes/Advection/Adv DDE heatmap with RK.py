## Solving DDE for heatmap with RK


##Press play esta noche maomeno 6 horas

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm
from Backup.code_backup import amp_values, calculate_amplitude_period
from itertools import combinations


def X_equation(x, y_delta, m, mu, Delta, a):
    return 1 / (1 + (np.exp(-mu*Delta/a) * y_delta)**m) - x

def Y_equation(x, y, k1, k2, K, m, q1, q2):
    return ((k1 * k2 /(K ** (m+1) * q1 * q2)) * x - y) / (q1 / q2)


K = 1.0
m = 5.0
k1 = 0.1
k2 = 0.1
q1 = 0.03
q2 = 0.03
a = 0.35                     # Advection speed
mu = 0.03
Delta = 7.5


a_values = [0.1, 0.1125, 0.125, 0.15, 0.35, 0.3625, 0.375]
m_values = [4.8, 4.9]
q1_values = np.linspace(0.01, 0.1, 25)
q2_values = np.linspace(0.01, 0.1, 25)
k1_values = np.linspace(0.05, 0.15, 25)
k2_values = np.linspace(0.05, 0.5, 25)
K_values = [0.95, 1, 1.05, 1.1, 1.15]
Delta_values = [7, 7.05, 7.1, 7.15, 7.2, 7.25]
mu_values = [0.0]

parameters = [k1_values, k2_values, q1_values, q2_values]
parameter_names = ['k1_values', 'k2_values', 'q1_values', 'q2_values']

combs = list(combinations(range(len(parameters)), 2))

t_span = (0, 200)           
dt = 0.0005                    # Si cambiamos el step, hay que cambiar el round(tau, 2)
t_steps = int(t_span[-1]/dt) 

##como traducir t_steps a segundos?

X = np.zeros([t_steps])
Y = np.zeros([t_steps])

time = np.linspace(0,t_span[-1],t_steps)   

# amplitud_final = np.zeros_like(q2_values)
# periodo_final = np.zeros_like(amplitud_final)

amplitude_map = []
period_map = []

fig, axs = plt.subplots(2, 3, figsize=(14, 6))

for k, (idx1, idx2) in enumerate(combs):
    k1 = 0.1
    k2 = 0.1
    q1 = 0.03
    q2 = 0.03
    q1_values = np.linspace(0.01, 0.2, 30)
    q2_values = np.linspace(0.01, 0.2, 30)
    k1_values = np.linspace(0.05, 0.15, 30)
    k2_values = np.linspace(0.05, 0.5, 30)
    amplitude_map = []
    period_map = []
    if idx1 == 0:
        if idx2 == 1:
            k1_values = np.linspace(0.05, 0.5, 30)
            for k1, k2 in tqdm(product(k1_values, k2_values)):
                params = {'k1': k1, 'k2': k2}
                p1_values = k1_values
                p2_values = k2_values
                X[0] = (q1 * K**m) / k1
                Y[0] = 0
                tau = q1 * Delta / a
                tau = round(tau, 4)
                for i in tqdm(range(1,t_steps)):
                    Yprime = Y[int(i - 1 - tau/dt)] if i - tau/dt > 0 else Y[0]
                    k1_x = X_equation(X[i-1], Yprime, m, mu, Delta, a)
                    k1_y = Y_equation(X[i-1], Y[i-1], k1, k2, K, m, q1, q2)
                    k1_y_prime = Y_equation(X[int(i-1 - tau/dt)], Y[int(i-1-tau/dt)], k1, k2, K, m, q1, q2) if i - tau/dt > 0 else Y[0]

                    aux_x = X[i-1]+dt*k1_x
                    aux_y = Y[i-1]+dt*k1_y
                    aux_y_prime = Yprime+dt*k1_y_prime if i- tau/dt > 0 else Y[0]
                    ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
                    k2_x = X_equation(aux_x, aux_y_prime, m, mu, Delta, a)
                    k2_y = Y_equation(aux_x, aux_y, k1, k2, K, m, q1, q2)
        
                    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                    Y[i] = Y[i-1]+0.5*dt*(k1_y+k2_y)
                amplitude, period = calculate_amplitude_period(time, X[:])
                amplitude_map.append(amplitude)
                period_map.append(period)
            amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
            period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

            im1 = axs[0, 0].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
            axs[0, 0].set_title('k1 vs k2')
            axs[0, 0].set_xlabel('k2')
            axs[0, 0].set_ylabel('k1')
            fig.colorbar(im1, ax=axs[0, 0], label='Amplitude')
        elif idx2 == 2:
            for k1, q1 in tqdm(product(k1_values, q1_values)):
                params = {'k1': k1, 'q1': q1}
                p1_values = k1_values
                p2_values = q1_values
                X[0] = (q1 * K**m) / k1
                Y[0] = 0
                tau = q1 * Delta / a
                tau = round(tau, 4)
                for i in tqdm(range(1,t_steps)):
                    Yprime = Y[int(i - 1 - tau/dt)] if i - tau/dt > 0 else Y[0]
                    k1_x = X_equation(X[i-1], Yprime, m, mu, Delta, a)
                    k1_y = Y_equation(X[i-1], Y[i-1], k1, k2, K, m, q1, q2)
                    k1_y_prime = Y_equation(X[int(i-1 - tau/dt)], Y[int(i-1-tau/dt)], k1, k2, K, m, q1, q2) if i - tau/dt > 0 else Y[0]

                    aux_x = X[i-1]+dt*k1_x
                    aux_y = Y[i-1]+dt*k1_y
                    aux_y_prime = Yprime+dt*k1_y_prime if i- tau/dt > 0 else Y[0]
                    ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
                    k2_x = X_equation(aux_x, aux_y_prime, m, mu, Delta, a)
                    k2_y = Y_equation(aux_x, aux_y, k1, k2, K, m, q1, q2)
        
                    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                    Y[i] = Y[i-1]+0.5*dt*(k1_y+k2_y)
                amplitude, period = calculate_amplitude_period(time, X[:])
                amplitude_map.append(amplitude)
                period_map.append(period)
            amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
            period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

            im1 = axs[0, 1].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
            axs[0, 1].set_title('k1 vs q1')
            axs[0, 1].set_xlabel('q1')
            axs[0, 1].set_ylabel('k1')
            fig.colorbar(im1, ax=axs[0, 1], label='Amplitude')
        elif idx2 == 3:
            for k1, q2 in tqdm(product(k1_values, q2_values)):
                params = {'k1': k1, 'q2': q2}
                p1_values = k1_values
                p2_values = q2_values
                X[0] = (q1 * K**m) / k1
                Y[0] = 0
                tau = q1 * Delta / a
                tau = round(tau, 4)
                for i in tqdm(range(1,t_steps)):
                    Yprime = Y[int(i - 1 - tau/dt)] if i - tau/dt > 0 else Y[0]
                    k1_x = X_equation(X[i-1], Yprime, m, mu, Delta, a)
                    k1_y = Y_equation(X[i-1], Y[i-1], k1, k2, K, m, q1, q2)
                    k1_y_prime = Y_equation(X[int(i-1 - tau/dt)], Y[int(i-1-tau/dt)], k1, k2, K, m, q1, q2) if i - tau/dt > 0 else Y[0]

                    aux_x = X[i-1]+dt*k1_x
                    aux_y = Y[i-1]+dt*k1_y
                    aux_y_prime = Yprime+dt*k1_y_prime if i- tau/dt > 0 else Y[0]
                    ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
                    k2_x = X_equation(aux_x, aux_y_prime, m, mu, Delta, a)
                    k2_y = Y_equation(aux_x, aux_y, k1, k2, K, m, q1, q2)
        
                    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                    Y[i] = Y[i-1]+0.5*dt*(k1_y+k2_y)
                amplitude, period = calculate_amplitude_period(time, X[:])
                amplitude_map.append(amplitude)
                period_map.append(period)
            amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
            period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

            im1 = axs[0, 2].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
            axs[0, 2].set_title('k1 vs q2')
            axs[0, 2].set_xlabel('q2')
            axs[0, 2].set_ylabel('k1')
            fig.colorbar(im1, ax=axs[0, 2], label='Amplitude')
    elif idx1 == 1:
        if idx2 == 2:
            k2_values = np.linspace(0.01, 0.8, 30)
            q1_values = np.linspace(0.01, 0.3, 30)
            for k2, q1 in tqdm(product(k2_values, q1_values)):
                params = {'k2': k2, 'q1': q1}
                p1_values = k2_values
                p2_values = q1_values
                X[0] = (q1 * K**m) / k1
                Y[0] = 0
                tau = q1 * Delta / a
                tau = round(tau, 4)
                for i in tqdm(range(1,t_steps)):
                    Yprime = Y[int(i - 1 - tau/dt)] if i - tau/dt > 0 else Y[0]
                    k1_x = X_equation(X[i-1], Yprime, m, mu, Delta, a)
                    k1_y = Y_equation(X[i-1], Y[i-1], k1, k2, K, m, q1, q2)
                    k1_y_prime = Y_equation(X[int(i-1 - tau/dt)], Y[int(i-1-tau/dt)], k1, k2, K, m, q1, q2) if i - tau/dt > 0 else Y[0]

                    aux_x = X[i-1]+dt*k1_x
                    aux_y = Y[i-1]+dt*k1_y
                    aux_y_prime = Yprime+dt*k1_y_prime if i- tau/dt > 0 else Y[0]
                    ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
                    k2_x = X_equation(aux_x, aux_y_prime, m, mu, Delta, a)
                    k2_y = Y_equation(aux_x, aux_y, k1, k2, K, m, q1, q2)
        
                    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                    Y[i] = Y[i-1]+0.5*dt*(k1_y+k2_y)
                amplitude, period = calculate_amplitude_period(time, X[:])
                amplitude_map.append(amplitude)
                period_map.append(period)
            amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
            period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

            im1 = axs[1, 0].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
            axs[1, 0].set_title('k2 vs q1')
            axs[1, 0].set_xlabel('q1')
            axs[1, 0].set_ylabel('k2')
            fig.colorbar(im1, ax=axs[1, 0], label='Amplitude')
        elif idx2 == 3:
            k2_values = np.linspace(0.01, 0.8, 30)
            q2_values = np.linspace(0.01, 0.3, 30)
            for k2, q2 in tqdm(product(k2_values, q2_values)):
                params = {'k2': k2, 'q2': q2}
                p1_values = k2_values
                p2_values = q2_values
                X[0] = (q1 * K**m) / k1
                Y[0] = 0
                tau = q1 * Delta / a
                tau = round(tau, 4)
                for i in tqdm(range(1,t_steps)):
                    Yprime = Y[int(i - 1 - tau/dt)] if i - tau/dt > 0 else Y[0]
                    k1_x = X_equation(X[i-1], Yprime, m, mu, Delta, a)
                    k1_y = Y_equation(X[i-1], Y[i-1], k1, k2, K, m, q1, q2)
                    k1_y_prime = Y_equation(X[int(i-1 - tau/dt)], Y[int(i-1-tau/dt)], k1, k2, K, m, q1, q2) if i - tau/dt > 0 else Y[0]

                    aux_x = X[i-1]+dt*k1_x
                    aux_y = Y[i-1]+dt*k1_y
                    aux_y_prime = Yprime+dt*k1_y_prime if i- tau/dt > 0 else Y[0]
                    ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
                    k2_x = X_equation(aux_x, aux_y_prime, m, mu, Delta, a)
                    k2_y = Y_equation(aux_x, aux_y, k1, k2, K, m, q1, q2)
        
                    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                    Y[i] = Y[i-1]+0.5*dt*(k1_y+k2_y)
                amplitude, period = calculate_amplitude_period(time, X[:])
                amplitude_map.append(amplitude)
                period_map.append(period)
            amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
            period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

            im1 = axs[1, 1].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
            axs[1, 1].set_title('k2 vs q2')
            axs[1, 1].set_xlabel('q2')
            axs[1, 1].set_ylabel('k2')
            fig.colorbar(im1, ax=axs[1, 1], label='Amplitude')
    else:
        q1_values = np.linspace(0.02, 0.15, 30)
        q2_values = np.linspace(0.02, 0.15, 30)
        for q1, q2 in tqdm(product(q1_values, q2_values)):
            params = {'q1': q1, 'q2': q2}
            p1_values = q1_values
            p2_values = q2_values
            X[0] = (q1 * K**m) / k1
            Y[0] = 0
            tau = q1 * Delta / a
            tau = round(tau, 4)
            for i in tqdm(range(1,t_steps)):
                Yprime = Y[int(i - 1 - tau/dt)] if i - tau/dt > 0 else Y[0]
                k1_x = X_equation(X[i-1], Yprime, m, mu, Delta, a)
                k1_y = Y_equation(X[i-1], Y[i-1], k1, k2, K, m, q1, q2)
                k1_y_prime = Y_equation(X[int(i-1 - tau/dt)], Y[int(i-1-tau/dt)], k1, k2, K, m, q1, q2) if i - tau/dt > 0 else Y[0]

                aux_x = X[i-1]+dt*k1_x
                aux_y = Y[i-1]+dt*k1_y
                aux_y_prime = Yprime+dt*k1_y_prime if i- tau/dt > 0 else Y[0]
                    ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
                k2_x = X_equation(aux_x, aux_y_prime, m, mu, Delta, a)
                k2_y = Y_equation(aux_x, aux_y, k1, k2, K, m, q1, q2)
        
                X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                Y[i] = Y[i-1]+0.5*dt*(k1_y+k2_y)
            amplitude, period = calculate_amplitude_period(time, X[:])
            amplitude_map.append(amplitude)
            period_map.append(period)
        amplitude_map_res= np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
        period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

        im1 = axs[1, 2].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
        axs[1, 2].set_title('q1 vs q2')
        axs[1, 2].set_xlabel('q2')
        axs[1, 2].set_ylabel('q1')
        fig.colorbar(im1, ax=axs[1, 2], label='Amplitude')
        
plt.tight_layout()
plt.show()



# for k in tqdm(range(0, len)):
#     for j in tqdm(range(0, len(q2_values))):
#         q2 = q2_values[j]
#         X[0, j] = 1
#         Y[0, j] = 0
#         tau = q1 * Delta / a
#         tau = round(tau, 3)
#         for i in tqdm(range(1,t_steps)):
#             Yprime = Y[int(i - 1 - tau/dt), j] if i - tau/dt > 0 else Y[0, j]
#             k1_x = X_equation(X[i-1, j], Yprime, m, mu, Delta, a)
#             k1_y = Y_equation(X[i-1, j], Y[i-1, j], k1, k2, K, m, q1, q2)
#             k1_y_prime = Y_equation(X[int(i-1 - tau/dt), j], Y[int(i-1-tau/dt), j], k1, k2, K, m, q1, q2) if i - tau/dt > 0 else Y[0, j]

#             aux_x = X[i-1, j]+dt*k1_x
#             aux_y = Y[i-1, j]+dt*k1_y
#             aux_y_prime = Yprime+dt*k1_y_prime if i- tau/dt > 0 else Y[0, j]
#         ##aux_y_pri = Y[int(i-1 - tau/dt)]+dt*k1_y if i - tau/dt > 0 else Y[0, j]
        
#             k2_x = X_equation(aux_x, aux_y_prime, m, mu, Delta, a)
#             k2_y = Y_equation(aux_x, aux_y, k1, k2, K, m, q1, q2)
        
#             X[i, j] = X[i-1, j]+0.5*dt*(k1_x+k2_x)
#             Y[i, j] = Y[i-1, j]+0.5*dt*(k1_y+k2_y)
#         amplitud_final[j], periodo_final[j] = calculate_amplitude_period(time, X[:, j])

# for j in tqdm(range(0, len(mu_values))):
#     plt.plot(time, X[:, j], label=f'mu ={mu_values[j]}')

# plt.xlabel("l")
# plt.ylabel("x(l)")
# plt.title("x(l) solution")
# plt.legend()
# plt.grid()


# for j in tqdm(range(0, len(q2_values))):
#     amplitud_final[j] = calculate_amplitude_period(time, X[:, j])


# plt.plot(time, X[:, 0])
# #plt.plot(q2_values, amplitud_final)

# plt.xlabel("q2")
# plt.ylabel("Amplitude")
# plt.title("hopf bifurcation")
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()