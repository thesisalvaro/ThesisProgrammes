##Esto es una prueba para resolver Goodwin model de mierda usando múltiples parámetros (cuando digo múltiples
## me refiero a tela) así que vamo a darle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from ddeint import ddeint
from itertools import product
import seaborn as sn

#tau = 0.56
p = 1
#m = 5
b = 11.11
#a_values = [0.405, 0.810, 1.215]
q = 0.03
Delta = 7.5

def goodwin(Y, l, params):
    K = params['K']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    q1 = params['q1']
    q2 = params['q2']
    q3 = params['q3']
    x = Y(l)[0]
    y = Y(l)[1]
    z = Y(l)[2]
    dxdl = k1 / (K**m + z**m) - q1 * x
    dydl = k2 * x - q2 * y
    dzdl = k3 * y - q3 * z
    return np.array([dxdl, dydl, dzdl])


def adv1(Y, l, params):
    a = params['a']
    x = Y(l)[0]
    y = Y(l)[1]
    tau = q * Delta / a
    x_retraso = Y(l - tau)[1] if l - tau > 0 else Y(0)[1]    
    dxdl = 1 / (1 + x_retraso**m) - x
    dydl = (b * x - y) / p
    return np.array([dxdl, dydl])


m_values = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  #hasta 7
a_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,]  #hasta 0.6
q1_values = [0.03]
q2_values = [0.03]
Delta_values = [7.5]
k3_values = [100]

## we have to transform these values to the adimensionalizated system

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

l_span = (0, 100)
l_eval = np.linspace(*l_span, 1001)
t_eval = np.linspace(*l_span, 100)
x0 = 1
y0 = 0
z0 = 0
y0_func = lambda l: np.array([x0, y0])
y0_func2 = lambda l: np.array([x0, y0, z0])

#print(l_eval[0], l_eval[100], l_eval[200], l_eval[300], l_eval[600], l_eval[1000])
#print(np.array([x0, y0]))

plt.figure(figsize=(10, 6))
#for a in a_values:
#    params = {'a': a}
 #   sol = ddeint(adv1, y0_func, l_eval, fargs=(params,))
    #sol = stm(adv1, y0_func, l_eval, fargs=(params,))

  #  plt.plot(l_eval, sol[:, 0], label=f"x(l), a={a}")
   # plt.plot(l_eval, sol[:, 1], label=f"y(l), a={a}", linestyle='dashed')

heatmap_data = []
for a, m, q1, q2, Delta, k3 in product(a_values, m_values, q1_values, q2_values, Delta_values, k3_values):
    params = {'a': a, 'm': m, 'q1': q1, 'q2':q2, 'Delta': Delta, 'k3': k3}
    sol = ddeint(adv2, y0_func, l_eval, fargs=(params,))
    tau = q * Delta / a
    plt.plot(l_eval, sol[:, 0], label=f"x(l), a={a}, m = {m}, p={p}, b={b}, q={q}, Delta={Delta}, tau={round(tau, 2)}")
    heatmap_data.append(sol[:, 0])  ##This shit here takes sol.x and appends to heatmap so it only gets the value of x for the 10k time values
    #plt.plot(l_eval, sol[:, 1], label=f"y(l), a={a}, m = {m}", linestyle='dashed')

plt.title("Solutions")
plt.xlabel("l")
plt.ylim(0, 0.45)
plt.ylabel("Values of x(l)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

print('size is', np.size(heatmap_data), 'shape is ', np.shape(heatmap_data), 'dim is', np.ndim(heatmap_data))

times = [0, 100, 200, 300, 600, 1000] # To see whatever happens here hahahahaha idontknowwhat am i doin wit ma life

#for index in zip(len(a_values), len(m_values)):
heatmap_data = np.array(heatmap_data)

print('size is', np.size(heatmap_data), 'shape is ', np.shape(heatmap_data), 'dim is', np.ndim(heatmap_data))

hm = []
#for i in enumerate(times):
 #   hm.append(heatmap_data[:,i])

#print('size is', np.size(hm), 'shape is ', np.shape(hm), 'dim is', np.ndim(hm))

#print(hm)

num_a = 9
num_m = 9

hm = np.array(hm)

fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

for i, t_idx in enumerate(times):
    hm_reshaped = heatmap_data[:, t_idx].reshape(num_a, num_m)
    # plt.figure(figsize=(8, 6))
    # plt.imshow(hm_reshaped, cmap='viridis', aspect='auto')
    # plt.colorbar(label=f'Value at t={t_idx}')
    # plt.title(f'Heatmap of Parameter a vs. Parameter m at t={t_idx}')
    # plt.xlabel('Parameter m Index')
    # plt.ylabel('Parameter a Index')
    # plt.show()
    ax = axes[i // 3, i % 3]  # Map i to a 2x3 grid of subplots
    
    # Plot the heatmap
    im = ax.imshow(hm_reshaped, cmap='viridis', aspect='auto')
    ax.set_title(f't = {t_idx}')
    ax.set_xlabel('Parameter m Index')
    ax.set_ylabel('Parameter a Index')

# Add a single colorbar for all heatmaps
fig.colorbar(im, ax=axes, location='right', shrink=0.7, label='Value')

# Set a super title for the entire figure
fig.suptitle('Heatmaps of Parameter a vs. Parameter m at Selected Times', fontsize=16)

# Show the plot
plt.show()

print('size is', np.size(hm_reshaped), 'shape is ', np.shape(hm_reshaped), 'dim is', np.ndim(hm_reshaped))

print(hm_reshaped)

exit()
#version con 6 plots uno detrás de otro

# for i in range(np.shape(hm)[0]):  # Loop over the first dimension (6 slices)
#     plt.figure(figsize=(6, 4))  # Create a new figure for each heatmap
#     plt.title(f"Heatmap {i + 1}")  # Title for the heatmap
#     plt.imshow(hm[i], cmap='viridis', aspect='auto')  # Heatmap for the i-th slice
#     plt.colorbar(label='Values')  # Add a colorbar to indicate value scales
#     plt.xlabel("Column Index")  # Label for the x-axis
#     plt.ylabel("Row Index")  # Label for the y-axis
#     plt.show()  # Display the heatmap


# exit()
#version con 6 subplots del tirón

# fig, axs = plt.subplots(1, 6, figsize=(15, 15))
# for i in range(np.shape(hm)[0]):  # Loop over the first dimension (6 slices)
#     #plt.figure(figsize=(6, 4))  # Create a new figure for each heatmap
#     #plt.title(f"Heatmap {i + 1}")  # Title for the heatmap
#     axs[i].imshow(hm[i], cmap='viridis', aspect='auto')  # Heatmap for the i-th slice
#     axs[i]._colorbars(label='Values')  # Add a colorbar to indicate value scales
#     #plt.xlabel("Column Index")  # Label for the x-axis
#     #plt.ylabel("Row Index")  # Label for the y-axis
#     #plt.show()  # Display the heatmap

# plt.show()
# exit()


#print(heatmap_data)

#exit()

#print(heatmap_data)
#hm = sn.heatmap(data = heatmap_data)
#plt.show()

#exit()



# fig, axs = plt.subplots(2, 5, figsize=(8, 8))
# #axs[0,0]sn.heatmap(heatmap_data[])
# axs[0,0].grid(True)

# exit()

# axs[0,1].plot(t_eval, x)
# axs[0,1].grid(True)

# axs[0,2].plot(t_eval, x)
# axs[0,2].grid(True)

# axs[0,3].plot(t_eval, x)
# axs[0,3].grid(True)

# axs[0,4].plot(t_eval, x)
# axs[0,4].grid(True)

# axs[1,0].plot(t_eval, x)
# axs[1,0].grid(True)

# axs[1,1].plot(t_eval, x)
# axs[1,1].grid(True)

# axs[1,2].plot(t_eval, x)
# axs[1,2].grid(True)

# axs[1,3].plot(t_eval, x)
# axs[1,3].grid(True)

# axs[1,4].plot(t_eval, x)
# axs[1,4].grid(True)



# t = l_eval
# X = sol[0, :]
# Y = sol[1:, :]    



# #exit()

# ##heatmap adsdjsiadjaidjsaidj
# plt.figure(figsize=(10, 6))
# plt.imshow(heatmap_data, extent=[m_values[0], m_values[1], a_values[0], a_values[-1]], aspect='auto', origin='lower', cmap='viridis')
# plt.colorbar(label='x(l)')
# plt.xlabel('Time-like variable (l)')
# plt.ylabel('Parameter a')
# plt.title('Heatmap of x(l) for varying a and l')
# plt.show()

# exit()

# plt.figure(figsize=(10, 6))
# plt.imshow(heatmap_data, extent=[m_values[0], m_values[-1], a_values[0], a_values[-1]], aspect='auto', origin='lower', cmap='viridis')
# plt.colorbar(label='x(l)')
# plt.xlabel('Parameter m')
# plt.ylabel('Parameter a')
# plt.title('Amplitude Heatmap of x(l) in (a, m) Phase Space')
# plt.show()

##Mathematical delayed systems can create oscillations --> how?? 
