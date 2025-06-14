##advPDEdegheatmap
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

# Ready pa usarse esta noche

# Define the function to calculate amplitude and period
def calculate_amplitude_period(time, values):
    maxs = []
    maxs_times = []
    
    mins = []
    mins_times = []

    # Identify local maxima
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxs.append(values[i])
            maxs_times.append(time[i])

    #take sum min too lol
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            mins.append(values[i])
            mins_times.append(time[i])

    if len(maxs) < 2:
        return 0, 0  # Not enough oscillations to compute period

    if len(mins) < 2:
        return 0, 0  # Not enough oscillations to compute period

    amplitude = (maxs[-1] - mins[-1] ) / 2
    if maxs_times[-1] > mins_times[-1]:
        period = maxs_times[-1] - maxs_times[-2]
    else:
        period = mins_times[-1] - mins_times[-2]

    if amplitude <= 0.00001:
        return amplitude, 0
        
    return amplitude, period

#Parameters
m_values = np.linspace(3, 7, 5) #de 3 a 7
a_values = np.linspace(0.1, 0.7, 33)  #de 0.2 a 0.6
q1_values = np.linspace(0.01, 0.1, 5)
q2_values = np.linspace(0.01, 0.1, 5)
Delta_values = [7.5]
k3_values = [100]
q_values = [0.03]
mu_values = np.linspace(0.01, 0.05, 33)

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
a = 0.405                     # Advection speed
mu = 0.03
Delta = 7.5

# Grid
C = 100.0                   # Spatial domain limit
Nx = 1001                   # Number of spatial points
x = np.linspace(0, C, Nx)   # Spatial grid
dx = x[1] - x[0]            # Spatial step

                 # Position for Y(x = Delta)
delta_idx = np.abs(x-Delta).argmin()

# Time span
t_span = (0, 5000)           # Time range for simulation
dt = 0.01                    # Time step for numerical evaluation
t_steps = int(t_span[-1]/dt) 
t_eval = np.linspace(0, 500, t_steps)

# Initial conditions
X = np.ones(t_steps)     # Initial value for X
Y = np.zeros([Nx,t_steps]) # Initial values for Y(x, t)
#Y0 = np.zeros([t_steps])
#Y0[0] = 0

time = np.linspace(0,t_span[-1],t_steps) 

p1_values = a_values
p2_values = mu_values

amplitude_map = []
period_map = []


for a, mu in tqdm(product(a_values, mu_values)):
    params = {'a': a, 'mu': mu}
    for i in tqdm(range(1,t_steps)):
        k1_x = X_equation(X[i-1], Y[delta_idx,i-1], k1, K, m, q1)
        #k1_y0 = y0_equation(X[i-1], Y0[i-1], k2, q2)
        k1_y = Y_equation((i-1)*dt, X[i-1], Y[:,i-1].copy(), k2, q2, a, dx, mu)
        

        aux_x = X[i-1]+dt*k1_x
        #aux_y0 = Y0[i-1] + dt*k1_y0
        aux_y = Y[:,i-1]+dt*k1_y
        
        
        k2_x = X_equation(aux_x, aux_y[delta_idx], k1, K, m, q1)
        #k2_y0 = y0_equation(aux_x, aux_y0, k2, q2)
        k2_y = Y_equation(i*dt, aux_x, aux_y, k2, q2, a, dx, mu)
        
        X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
        #Y0[i] = Y0[i-1] + 0.5*dt*(k1_y0+k2_y0)
        Y[:,i] = Y[:,i-1]+0.5*dt*(k1_y+k2_y)

    #plt.plot(l_eval, sol[:, 0], label=f"x(l), a={a}, m = {m}, p={p}, b={round(b, 2)}, q1={q1}, q2={q2}, Delta={Delta}, tau={round(tau, 2)}, K*={k3}") #This here plots x(l) ye
    amplitude, period = calculate_amplitude_period(t_eval, X[:])
    amplitude_map.append(amplitude)
    period_map.append(period)
    #plt.plot(l_eval, sol[:, 1], label=f"y(l), a={a}, m = {m}", linestyle='dashed') #This here actually adds the y(x,t), so no interested rn



amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))


# Plot heatmaps
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap for amplitude
im1 = axs[0].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
axs[0].set_title('Amplitude Heatmap')
axs[0].set_xlabel('μ')
axs[0].set_ylabel('a')
fig.colorbar(im1, ax=axs[0], label='Amplitude')

# Heatmap for period
im2 = axs[1].imshow(period_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='plasma')
axs[1].set_title('Period Heatmap')
axs[1].set_xlabel('μ')
axs[1].set_ylabel('a')
fig.colorbar(im2, ax=axs[1], label='Period')

plt.tight_layout()
plt.show()


# Parameters
# m_values = np.linspace(3, 7, 5)  # Adjust range and step as needed
# a_values = np.linspace(0.2, 0.6, 5)  # Adjust range and step as needed
# q1 = 0.03
# q2 = 0.02
# Delta = 7.5
# k3 = 100
# l_span = (0, 50)
# l_eval = np.linspace(*l_span, 10001)
# x0 = 1
# y0 = 0
# y0_func = lambda l: np.array([x0, y0])

# Full parametrized DDE function
# def adv2(Y, l, params):
#     a = params['a']
#     q1 = params['q1']
#     q2 = params['q2']
#     Delta = params['Delta']
#     m = params['m']
#     k3 = params['k3']
#     b = 1 / (k3 * q1 * q2)
#     tau = q1 * Delta / a
#     p = q1 / q2
#     x = Y(l)[0]
#     y = Y(l)[1]
#     x_retraso = Y(l - tau)[1] if l - tau > 0 else Y(0)[1]
#     dxdl = 1 / (1 + x_retraso**m) - x
#     dydl = (b * x - y) / p
#     return np.array([dxdl, dydl])

#Here we put whatever parameter we want in the heatmap, eg a and m

# p1_values = ad.a_values
# p2_values = ad.m_values

# # Initialize arrays to store results
# amplitude_map = np.zeros((len(p1_values), len(p2_values)))
# period_map = np.zeros((len(p1_values), len(p2_values)))

# # Iterate over parameter combinations
# for i, a in enumerate(p1_values):
#     for j, m in enumerate(p2_values):
#         params = {'a': a, 'm': m, 'q1': ad.q1, 'q2': ad.q2, 'Delta': ad.Delta, 'k3': ad.k3}
#         sol = ddeint(ad.adv2, ad.y0_func, ad.l_eval, fargs=(params,))

#         # Calculate amplitude and period using the first variable (x)
#         amplitude, period = calculate_amplitude_period(ad.l_eval, sol[:, 0])

#         if amplitude is not None and period is not None:
#             amplitude_map[i, j] = amplitude
#             period_map[i, j] = period

# # Plot heatmaps
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# # Heatmap for amplitude
# im1 = axs[0].imshow(amplitude_map, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
# axs[0].set_title('Amplitude Heatmap')
# axs[0].set_xlabel('m')
# axs[0].set_ylabel('a')
# fig.colorbar(im1, ax=axs[0], label='Amplitude')

# # Heatmap for period
# im2 = axs[1].imshow(period_map, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='plasma')
# axs[1].set_title('Period Heatmap')
# axs[1].set_xlabel('m')
# axs[1].set_ylabel('a')
# fig.colorbar(im2, ax=axs[1], label='Period')

# plt.tight_layout()
# plt.show()
