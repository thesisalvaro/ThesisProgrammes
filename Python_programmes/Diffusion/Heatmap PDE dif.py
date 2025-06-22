## Solving Diffusion DDE for heatmap with RK

##Press play (takes around the whole night), be careful there are two different loops for the deg and no deg cases

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
mu = 0.03            ##DEGRADATION DEPENDS ON WHICH CASE
Delta = 4.5

##PARAMETERS TO VARY ----------------------------------------

a_values = np.linspace(0.4, 0.425, 6)
m_values = np.linspace(7, 15, 160)
q1_values = [10, 12, 2, 2, 2, 2]
q2_values = [0.015, 0.02, 0.025, 0.03, 0.13, 0.135, 0.14, 0.145]
k1_values = [0.05, 0.09, 0.095, 0.1]
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

t_span = (0, 3000)
dt = 0.01     

tau = np.zeros_like(Delta_values)


cfl_limit = (dx**2) / (2 * D)
if dt > cfl_limit:
    print(f"(dt = {dt}) > CFL limit ({cfl_limit:.5f}). Set dt to {cfl_limit:.5f}.")
    dt = cfl_limit


t_steps = int(t_span[-1]/dt) 


downsample_factor = 10  # Factor to downsample time steps for visualization
downsampled_steps = t_steps // downsample_factor
downsampled_steps = (t_steps + downsample_factor - 1) // downsample_factor

Y_full = None  # This will hold the full data during computation

Y_prev = np.zeros(Nx) 
Y_current = np.zeros(Nx)

X_downsampled = np.zeros((downsampled_steps))
time_downsampled = np.linspace(0, t_span[-1],downsampled_steps)


X = np.ones([t_steps])
time = np.linspace(0,t_span[-1],t_steps)
delta_idx = np.abs(x-Delta).argmin()

parameters = [k1_values, k2_values, q1_values, q2_values]
parameter_names = ['k1_values', 'k2_values', 'q1_values', 'q2_values']

combs = list(combinations(range(len(parameters)), 2))

amplitude_map = []
period_map = []

fig, axs = plt.subplots(2, 3, figsize=(14, 6))

##THIS ONE IS FOR mu = 0.03, SO WE HAVE TO SET mu = 0.03 TO RUN THIS
##UNCOMMENT THIS AND COMMENT THE ONE BELOW TO HAVE THE mu = 0.03 case

for k, (idx1, idx2) in enumerate(combs):
    k1 = 0.08
    k2 = 0.08
    q1 = 0.03
    q2 = 0.03
    q1_values = np.linspace(0.01, 0.15, 10)
    q2_values = np.linspace(0.01, 0.15, 10)
    k1_values = np.linspace(0.01, 0.5, 10)
    k2_values = np.linspace(0.01, 0.3, 10)
    amplitude_map = []
    period_map = []
    if idx1 == 0:
        if idx2 == 1:
            k1_values = np.linspace(0.01, 0.9, 25)
            k2_values = np.linspace(0.01, 0.6, 25)
            for k1, k2 in tqdm(product(k1_values, k2_values)):
                params = {'k1': k1, 'k2': k2}
                p1_values = k1_values
                p2_values = k2_values
                X[0] = 1
                Y_prev = np.zeros(Nx)
                for i in tqdm(range(1,t_steps)):
                    k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
                    k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
                    aux_x = X[i-1]+dt*k1_x
                    aux_y = Y_prev+dt*k1_y
        
                    k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
                    k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
                    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                    Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

        # Downsample data
                    if i % downsample_factor == 0:
                        downsample_index = i // downsample_factor
                        if downsample_index < downsampled_steps:
                            X_downsampled[downsample_index] = X[i]
                            #Y_downsampled[:, downsample_index, j] = Y_current

        # Update previous state
                    Y_prev = Y_current.copy()

                            
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
            k1_values = np.linspace(0.01, 0.8, 25)
            q1_values = np.linspace(0.01, 0.25, 25)
            for k1, q1 in tqdm(product(k1_values, q1_values)):
                params = {'k1': k1, 'q1': q1}
                p1_values = k1_values
                p2_values = q1_values
                X[0] = 1
                Y_prev = np.zeros(Nx)
                for i in tqdm(range(1,t_steps)):
                    k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
                    k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
                    aux_x = X[i-1]+dt*k1_x
                    aux_y = Y_prev+dt*k1_y
        
                    k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
                    k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
                    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                    Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

        # Downsample data
                    if i % downsample_factor == 0:
                        downsample_index = i // downsample_factor
                        if downsample_index < downsampled_steps:
                            X_downsampled[downsample_index] = X[i]
                            #Y_downsampled[:, downsample_index, j] = Y_current

        # Update previous state
                    Y_prev = Y_current.copy()


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
            k1_values = np.linspace(0.01, 0.8, 25)
            q2_values = np.linspace(0.01, 0.25, 25)
            for k1, q2 in tqdm(product(k1_values, q2_values)):
                params = {'k1': k1, 'q2': q2}
                p1_values = k1_values
                p2_values = q2_values
                X[0] = 1
                Y_prev = np.zeros(Nx)
                for i in tqdm(range(1,t_steps)):
                    k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
                    k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
                    aux_x = X[i-1]+dt*k1_x
                    aux_y = Y_prev+dt*k1_y
        
                    k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
                    k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
                    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                    Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

        # Downsample data
                    if i % downsample_factor == 0:
                        downsample_index = i // downsample_factor
                        if downsample_index < downsampled_steps: 
                            X_downsampled[downsample_index] = X[i]
                            #Y_downsampled[:, downsample_index, j] = Y_current

        # Update previous state
                    Y_prev = Y_current.copy()

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
            k2_values = np.linspace(0.01, 0.5, 25)
            q1_values = np.linspace(0.01, 0.2, 25)
            for k2, q1 in tqdm(product(k2_values, q1_values)):
                params = {'k2': k2, 'q1': q1}
                p1_values = k2_values
                p2_values = q1_values
                X[0] = 1
                Y_prev = np.zeros(Nx)
                for i in tqdm(range(1,t_steps)):
                    k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
                    k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
                    aux_x = X[i-1]+dt*k1_x
                    aux_y = Y_prev+dt*k1_y
        
                    k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
                    k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
                    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                    Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

        # Downsample data
                    if i % downsample_factor == 0:
                        downsample_index = i // downsample_factor
                        if downsample_index < downsampled_steps:
                            X_downsampled[downsample_index] = X[i]
                            #Y_downsampled[:, downsample_index, j] = Y_current

        # Update previous state
                    Y_prev = Y_current.copy()

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
            k2_values = np.linspace(0.01, 0.5, 25)
            q2_values = np.linspace(0.01, 0.25, 25)
            for k2, q2 in tqdm(product(k2_values, q2_values)):
                params = {'k2': k2, 'q2': q2}
                p1_values = k2_values
                p2_values = q2_values
                X[0] = 1
                Y_prev = np.zeros(Nx)
                for i in tqdm(range(1,t_steps)):
                    k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
                    k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
                    aux_x = X[i-1]+dt*k1_x
                    aux_y = Y_prev+dt*k1_y
        
                    k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
                    k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
                    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                    Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

        # Downsample data
                    if i % downsample_factor == 0:
                        downsample_index = i // downsample_factor
                        if downsample_index < downsampled_steps:
                            X_downsampled[downsample_index] = X[i]
                            #Y_downsampled[:, downsample_index, j] = Y_current

        # Update previous state
                    Y_prev = Y_current.copy()

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
        q1_values = np.linspace(0.01, 0.1, 25)
        q2_values = np.linspace(0.01, 0.1, 25)
        for q1, q2 in tqdm(product(q1_values, q2_values)):
            params = {'q1': q1, 'q2': q2}
            p1_values = q1_values
            p2_values = q2_values
            X[0] = 1
            Y_prev = np.zeros(Nx)
            for i in tqdm(range(1,t_steps)):
                k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
                k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
                aux_x = X[i-1]+dt*k1_x
                aux_y = Y_prev+dt*k1_y
        
                k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
                k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
                X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
                Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

        # Downsample data
                if i % downsample_factor == 0:
                    downsample_index = i // downsample_factor
                    if downsample_index < downsampled_steps:
                        X_downsampled[downsample_index] = X[i]
                        #Y_downsampled[:, downsample_index, j] = Y_current

        # Update previous state
                Y_prev = Y_current.copy()

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
        
#---------------------------------------------------------


##THIS ONE IS FOR THE CASE mu = 0, SO WE HAVE TO SET mu=0 in the parameters and then run for this one
##UNCOMMENT THIS AND COMMENT THE ONE ABOVE TO HAVE THE mu = 0 case



# for k, (idx1, idx2) in enumerate(combs):
#     k1 = 0.08
#     k2 = 0.08
#     q1 = 0.03
#     q2 = 0.03
#     q1_values = np.linspace(0.01, 0.12, 25)
#     q2_values = np.linspace(0.01, 0.12, 25)
#     k1_values = np.linspace(0.01, 0.3, 25)
#     k2_values = np.linspace(0.01, 0.3, 25)
#     amplitude_map = []
#     period_map = []
#     if idx1 == 0:
#         if idx2 == 1:
#             k1_values = np.linspace(0.01, 0.5, 25)
#             k2_values = np.linspace(0.01, 0.4, 25)
#             for k1, k2 in tqdm(product(k1_values, k2_values)):
#                 params = {'k1': k1, 'k2': k2}
#                 p1_values = k1_values
#                 p2_values = k2_values
#                 X[0] = 1
#                 Y_prev = np.zeros(Nx)
#                 for i in tqdm(range(1,t_steps)):
#                     k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
#                     k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
#                     aux_x = X[i-1]+dt*k1_x
#                     aux_y = Y_prev+dt*k1_y
        
#                     k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
#                     k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
#                     X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
#                     Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

#         # Downsample data
#                     if i % downsample_factor == 0:
#                         downsample_index = i // downsample_factor
#                         if downsample_index < downsampled_steps:  # Ensure within bounds
#                             X_downsampled[downsample_index] = X[i]
#                             #Y_downsampled[:, downsample_index, j] = Y_current

#         # Update previous state
#                     Y_prev = Y_current.copy()

                            
#                 amplitude, period = calculate_amplitude_period(time, X[:])
#                 amplitude_map.append(amplitude)
#                 period_map.append(period)
#             amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
#             period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

#             im1 = axs[0, 0].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
#             axs[0, 0].set_title('k1 vs k2')
#             axs[0, 0].set_xlabel('k2')
#             axs[0, 0].set_ylabel('k1')
#             fig.colorbar(im1, ax=axs[0, 0], label='Amplitude')
#         elif idx2 == 2:
#             k1_values = np.linspace(0.01, 0.5, 25)
#             q1_values = np.linspace(0.01, 0.15, 25)
#             for k1, q1 in tqdm(product(k1_values, q1_values)):
#                 params = {'k1': k1, 'q1': q1}
#                 p1_values = k1_values
#                 p2_values = q1_values
#                 X[0] = 1
#                 Y_prev = np.zeros(Nx)
#                 for i in tqdm(range(1,t_steps)):
#                     k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
#                     k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
#                     aux_x = X[i-1]+dt*k1_x
#                     aux_y = Y_prev+dt*k1_y
        
#                     k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
#                     k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
#                     X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
#                     Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

#         # Downsample data
#                     if i % downsample_factor == 0:
#                         downsample_index = i // downsample_factor
#                         if downsample_index < downsampled_steps:  # Ensure within bounds
#                             X_downsampled[downsample_index] = X[i]
#                             #Y_downsampled[:, downsample_index, j] = Y_current

#         # Update previous state
#                     Y_prev = Y_current.copy()


#                 amplitude, period = calculate_amplitude_period(time, X[:])
#                 amplitude_map.append(amplitude)
#                 period_map.append(period)
#             amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
#             period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

#             im1 = axs[0, 1].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
#             axs[0, 1].set_title('k1 vs q1')
#             axs[0, 1].set_xlabel('q1')
#             axs[0, 1].set_ylabel('k1')
#             fig.colorbar(im1, ax=axs[0, 1], label='Amplitude')
#         elif idx2 == 3:
#             k1_values = np.linspace(0.01, 0.5, 25)
#             q2_values = np.linspace(0.01, 0.15, 25)
#             for k1, q2 in tqdm(product(k1_values, q2_values)):
#                 params = {'k1': k1, 'q2': q2}
#                 p1_values = k1_values
#                 p2_values = q2_values
#                 X[0] = 1
#                 Y_prev = np.zeros(Nx)
#                 for i in tqdm(range(1,t_steps)):
#                     k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
#                     k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
#                     aux_x = X[i-1]+dt*k1_x
#                     aux_y = Y_prev+dt*k1_y
        
#                     k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
#                     k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
#                     X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
#                     Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

#         # Downsample data
#                     if i % downsample_factor == 0:
#                         downsample_index = i // downsample_factor
#                         if downsample_index < downsampled_steps:  # Ensure within bounds
#                             X_downsampled[downsample_index] = X[i]
#                             #Y_downsampled[:, downsample_index, j] = Y_current

#         # Update previous state
#                     Y_prev = Y_current.copy()

#                 amplitude, period = calculate_amplitude_period(time, X[:])
#                 amplitude_map.append(amplitude)
#                 period_map.append(period)
#             amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
#             period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

#             im1 = axs[0, 2].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
#             axs[0, 2].set_title('k1 vs q2')
#             axs[0, 2].set_xlabel('q2')
#             axs[0, 2].set_ylabel('k1')
#             fig.colorbar(im1, ax=axs[0, 2], label='Amplitude')
#     elif idx1 == 1:
#         if idx2 == 2:
#             k2_values = np.linspace(0.01, 0.3, 25)
#             q1_values = np.linspace(0.01, 0.15, 25)
#             for k2, q1 in tqdm(product(k2_values, q1_values)):
#                 params = {'k2': k2, 'q1': q1}
#                 p1_values = k2_values
#                 p2_values = q1_values
#                 X[0] = 1
#                 Y_prev = np.zeros(Nx)
#                 for i in tqdm(range(1,t_steps)):
#                     k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
#                     k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
#                     aux_x = X[i-1]+dt*k1_x
#                     aux_y = Y_prev+dt*k1_y
        
#                     k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
#                     k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
#                     X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
#                     Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

#         # Downsample data
#                     if i % downsample_factor == 0:
#                         downsample_index = i // downsample_factor
#                         if downsample_index < downsampled_steps:  # Ensure within bounds
#                             X_downsampled[downsample_index] = X[i]
#                             #Y_downsampled[:, downsample_index, j] = Y_current

#         # Update previous state
#                     Y_prev = Y_current.copy()

#                 amplitude, period = calculate_amplitude_period(time, X[:])
#                 amplitude_map.append(amplitude)
#                 period_map.append(period)
#             amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
#             period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

#             im1 = axs[1, 0].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
#             axs[1, 0].set_title('k2 vs q1')
#             axs[1, 0].set_xlabel('q1')
#             axs[1, 0].set_ylabel('k2')
#             fig.colorbar(im1, ax=axs[1, 0], label='Amplitude')
#         elif idx2 == 3:
#             k2_values = np.linspace(0.01, 0.3, 25)
#             q2_values = np.linspace(0.01, 0.15, 25)
#             for k2, q2 in tqdm(product(k2_values, q2_values)):
#                 params = {'k2': k2, 'q2': q2}
#                 p1_values = k2_values
#                 p2_values = q2_values
#                 X[0] = 1
#                 Y_prev = np.zeros(Nx)
#                 for i in tqdm(range(1,t_steps)):
#                     k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
#                     k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
#                     aux_x = X[i-1]+dt*k1_x
#                     aux_y = Y_prev+dt*k1_y
        
#                     k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
#                     k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
#                     X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
#                     Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

#         # Downsample data
#                     if i % downsample_factor == 0:
#                         downsample_index = i // downsample_factor
#                         if downsample_index < downsampled_steps:  # Ensure within bounds
#                             X_downsampled[downsample_index] = X[i]
#                             #Y_downsampled[:, downsample_index, j] = Y_current

#         # Update previous state
#                     Y_prev = Y_current.copy()

#                 amplitude, period = calculate_amplitude_period(time, X[:])
#                 amplitude_map.append(amplitude)
#                 period_map.append(period)
#             amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
#             period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

#             im1 = axs[1, 1].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
#             axs[1, 1].set_title('k2 vs q2')
#             axs[1, 1].set_xlabel('q2')
#             axs[1, 1].set_ylabel('k2')
#             fig.colorbar(im1, ax=axs[1, 1], label='Amplitude')
#     else:
#         q1_values = np.linspace(0.01, 0.1, 25)
#         q2_values = np.linspace(0.01, 0.1, 25)
#         for q1, q2 in tqdm(product(q1_values, q2_values)):
#             params = {'q1': q1, 'q2': q2}
#             p1_values = q1_values
#             p2_values = q2_values
#             X[0] = 1
#             Y_prev = np.zeros(Nx)
#             for i in tqdm(range(1,t_steps)):
#                 k1_x = X_equationPDE(X[i-1], Y_prev[delta_idx], k1, K, m, q1)
#                 k1_y = Y_equationPDE((i-1)*dt, X[i-1], Y_prev.copy(), k2, q2, D, dx, mu)
        
#                 aux_x = X[i-1]+dt*k1_x
#                 aux_y = Y_prev+dt*k1_y
        
#                 k2_x = X_equationPDE(aux_x, aux_y[delta_idx], k1, K, m, q1)
#                 k2_y = Y_equationPDE(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)
        
#                 X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
#                 Y_current = Y_prev+0.5*dt*(k1_y+k2_y)

#         # Downsample data
#                 if i % downsample_factor == 0:
#                     downsample_index = i // downsample_factor
#                     if downsample_index < downsampled_steps:  # Ensure within bounds
#                         X_downsampled[downsample_index] = X[i]
#                         #Y_downsampled[:, downsample_index, j] = Y_current

#         # Update previous state
#                 Y_prev = Y_current.copy()

#             amplitude, period = calculate_amplitude_period(time, X[:])
#             amplitude_map.append(amplitude)
#             period_map.append(period)
#         amplitude_map_res= np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
#         period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

#         im1 = axs[1, 2].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
#         axs[1, 2].set_title('q1 vs q2')
#         axs[1, 2].set_xlabel('q2')
#         axs[1, 2].set_ylabel('q1')
#         fig.colorbar(im1, ax=axs[1, 2], label='Amplitude')
        


plt.tight_layout()
plt.show()
