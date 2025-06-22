## Another heatmap solving script, never really used

##Helped to create the final script, does not really work

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from itertools import product
import Adv1 as av


hm_data = []

for a, m, q1, q2, Delta, k3 in product(av.a_values, av.m_values, av.q1_values, av.q2_values, av.Delta_values, av.k3_values):
    params = {'a': a, 'm': m, 'q1': q1, 'q2':q2, 'Delta': Delta, 'k3': k3}
    sol = ddeint(av.adv2, av.y0_func, av.l_eval, fargs=(params,))
    tau = q1 * Delta / a
    b = 1 / (k3 * q1 * q2)
    p = q1 / q2
    hm_data.append(sol[:, 0])  ##This here takes sol.x and appends to heatmap so it only gets the value of x for the 10k time values

hm_data = np.array(hm_data)

##print('size is', np.size(hm_data), 'shape is ', np.shape(hm_data), 'dim is', np.ndim(hm_data))

num_a = len(av.a_values)
num_m = len(av.m_values)
num_t = len(av.l_eval)
times = [0, 100, 200, 300, 600, 1000]  ##times at which I want to evaluate x(l), have to configure them manually yet

##print(len(av.l_eval))

fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

hm_type = input('a vs m / a vs l?')

if hm_type == str('a vs m'):
    for i, t_idx in enumerate(times):
        hm_reshaped = hm_data[:, t_idx].reshape(num_a, num_m)
        ax = axes[i // 3, i % 3]
        im = ax.imshow(hm_reshaped, cmap='viridis', aspect='auto')
        ax.set_title(f't = {t_idx}')
        ax.set_xlabel('Parameter m Index')
        ax.set_ylabel('Parameter a Index')
    fig.colorbar(im, ax=axes, location='right', shrink=0.7, label='Value')
    fig.suptitle('Heatmap of a against m', fontsize=16)
    plt.show()   
elif hm_type == str('a vs l'):
    for i in range(num_m):
        ax = axes[i // 3, i % 3]
        hm_data = hm_data.reshape(num_a, num_m, num_t)
        a_vs_t_data = hm_data.transpose(1, 0, 2).reshape(num_m, num_a, num_t).reshape(num_m, num_a * num_t)
        heatmap_data = a_vs_t_data[i].reshape(num_a, num_t)
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
        ax.set_title(f'Heatmap for m={i+1}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Parameter a Index')
    fig.colorbar(im, ax=axes, location='right', shrink=0.7, label='Value')
    fig.suptitle('Heatmap of a against m', fontsize=16)
    plt.show()
else:
    print('error??')


# fig.colorbar(im, ax=axes, location='right', shrink=0.7, label='Value')
# fig.suptitle('Heatmap of a against m', fontsize=16)
# plt.show()


