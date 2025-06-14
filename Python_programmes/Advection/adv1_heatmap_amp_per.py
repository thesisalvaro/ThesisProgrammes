import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from itertools import product
# import Adv1 as ad

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
        return None, None  # Not enough oscillations to compute period

    if len(mins) < 2:
        return None, None  # Not enough oscillations to compute period

    amplitude = (maxs[-1] - mins[-1] ) / 2
    if maxs_times[-1] > mins_times[-1]:
        period = maxs_times[-1] - maxs_times[-2]
    else:
        period = mins_times[-1] - mins_times[-2]
        
    return amplitude, period

#Parameters
m_values = [5]  #de 3 a 7
a_values = [0.405]  #de 0.2 a 0.6
q1_values = np.linspace(0.01, 0.05, 33)
q2_values = np.linspace(0.01, 0.05, 33)
Delta_values = [7.5]
k3_values = [100]
q_values = [0.03]

l_span = (0, 50)
l_eval = np.linspace(*l_span, 20001)
t_eval = np.linspace(*l_span, 100)
x0 = 1
y0 = 0
z0 = 0
y0_func = lambda l: np.array([x0, y0])
y0_func2 = lambda l: np.array([x0, y0, z0])


def adv2(Y, l, params):
    a = params['a']
    q1 = params['q1']
    q2 = params['q2']
    Delta = params['Delta']
    m = params['m']
    k3 = params['k3']
    b = 1 / (k3 * q1 * q2)
    tau = q1 * Delta / a
    p = q1 / q2
    x = Y(l)[0]
    y = Y(l)[1]
    x_retraso = Y(l - tau)[1] if l - tau > 0 else Y(0)[1]
    dxdl = 1 / (1 + x_retraso**m) - x
    dydl = (b * x - y) / p
    return np.array([dxdl, dydl])


plt.figure(figsize=(10, 6))

p1_values = q1_values
p2_values = q2_values

amplitude_map = []
period_map = []

for a, m, q1, q2, Delta, k3 in product(a_values, m_values, q1_values, q2_values, Delta_values, k3_values):
    params = {'a': a, 'm': m, 'q1': q1, 'q2':q2, 'Delta': Delta, 'k3': k3}
    sol = ddeint(adv2, y0_func, l_eval, fargs=(params,))
    tau = q1 * Delta / a
    b = 1 / (k3 * q1 * q2)
    p = q1 / q2
    # plt.plot(l_eval, sol[:, 0], label=f"x(l), a={a}, m = {m}, p={p}, b={round(b, 2)}, q1={q1}, q2={q2}, Delta={Delta}, tau={round(tau, 2)}, K*={k3}") #This here plots x(l) ye
    amplitude, period = calculate_amplitude_period(l_eval, sol[:, 0])
    amplitude_map.append(amplitude)
    period_map.append(period)
    #plt.plot(l_eval, sol[:, 1], label=f"y(l), a={a}, m = {m}", linestyle='dashed') #This here actually adds the y(x,t), so no interested rn

# plt.title("Solutions")
# plt.xlabel("l")
# plt.ylim(0, 0.5)
# plt.ylabel("Values of x(l)")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

amplitude_map_res = np.reshape(amplitude_map, (len(p1_values), len(p2_values)))
period_map_res = np.reshape(period_map, (len(p1_values), len(p2_values)))

print(amplitude_map_res, period_map_res)

# Plot heatmaps
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap for amplitude
im1 = axs[0].imshow(amplitude_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='viridis')
axs[0].set_title('Amplitude Heatmap')
axs[0].set_xlabel('q2')
axs[0].set_ylabel('q1')
fig.colorbar(im1, ax=axs[0], label='Amplitude')

# Heatmap for period
im2 = axs[1].imshow(period_map_res, extent=(p2_values[0], p2_values[-1], p1_values[0], p1_values[-1]), origin='lower', aspect='auto', cmap='plasma')
axs[1].set_title('Period Heatmap')
axs[1].set_xlabel('q2')
axs[1].set_ylabel('q1')
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
