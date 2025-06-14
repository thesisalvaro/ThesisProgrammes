# ##adv openloop

# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import product
# import matplotlib
# matplotlib.use('QtAgg')  # or 'QtAgg' if available
# from tqdm import tqdm
# from Backup.code_backup import amp_values, calculate_amplitude_period

# def Y_equation(t, y, a, dx, mu): 
#     y[1:] = -(a/dx)*(y[1:]-y[:-1])
#     if t < 2:
#         y[0] = np.sin(np.pi * t)
#     else:
#         y[0] = 0
#     # y[0] = np.sin(np.pi * t)
#     return y



# a = 0.35                     # Advection speed
# mu = 0.0




# a_values = np.linspace(0.4, 0.425, 6)
# mu_values = [0.0]

# # Grid
# C = 30.0                   # Spatial domain limit
# Nx = 3001                   # Number of spatial points
# x = np.linspace(0, C, Nx)  # Spatial grid
# dx = x[1] - x[0]            # Spatial step

#                  # Position for Y(x = Delta)


# # Time span
# t_span = (0, 90)           # Time range for simulation
# dt = 0.02                    # Time step for numerical evaluation
# t_steps = int(t_span[-1]/dt) 


# # Initial conditions
# X = np.ones([t_steps, len(mu_values)])     # Initial value for X
# Y = np.zeros([Nx, t_steps]) # Initial values for Y(x, t)


# time = np.linspace(0,t_span[-1],t_steps)   

# plt.figure(figsize=(10, 6))

# Y1 = np.zeros([len(mu_values), t_steps])

# #def initial_condition(t):
#  #   return np.maximum(0, -np.sin(np.pi*t))
# for j in tqdm(range(0, len(mu_values))):
#     mu = mu_values[j]
#     for i in tqdm(range(1,t_steps)):

#         k1_y = Y_equation((i-1)*dt, Y[:,i-1].copy(), a, dx, mu)

#         aux_y = Y[:,i-1]+dt*k1_y
        
#         k2_y = Y_equation(i*dt, aux_y, a, dx, mu)

#         Y[:,i] = Y[:,i-1]+0.5*dt*(k1_y+k2_y)


    

# colors = ['b', 'g', 'y', 'r', 'k']


# for t_idx in range(0, t_steps, int(t_steps / 6)):  # Plot at intervals
#     plt.plot(x, Y[:, t_idx], label=f"t={t_idx * dt:.2f}")



# plt.xlabel("t")
# plt.ylabel("Y(t)")
# plt.title("advection openloop")
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 1.0                  # Advection speed
omega = 0.5 * np.pi      # Angular frequency
T = (2*np.pi) / omega            # Time period of the sinusoidal source
x_min, x_max = 0, 10     # Spatial domain
Nx = 500                 # Number of spatial points
t_values = [2, 5, 8]  # Time values to plot
x = np.linspace(x_min, x_max, Nx)  # Spatial grid

# Define the sinusoidal pulse function
def sinusoidal_pulse(x, t):
    tau = t - x / a
    if 0 <= tau <= T / 2:  # Source emits only one period
        return np.sin(omega * tau)
    else:
        return 0.0

# Vectorize the sinusoidal_pulse function for array operations
sinusoidal_pulse_vec = np.vectorize(sinusoidal_pulse, excluded=['t'])

# Plot the solution at different times
plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
u = sinusoidal_pulse_vec(x, t_values[0])
plt.plot(x, u, label=f't = {t_values[0]:.2f}')
plt.xlabel("x")
plt.ylabel("Y(x, t)")
#plt.title("Traveling Sinusoidal Pulse (One Period)")
plt.legend()
plt.xticks([])
plt.yticks([])
#plt.axis('off')
plt.axhline(0, color='black', linewidth=1.25)
plt.axvline(0, color='black', linewidth=1.25)
#plt.grid(True)

plt.subplot(1, 3, 2)
u = sinusoidal_pulse_vec(x, t_values[1])
plt.plot(x, u, label=f't = {t_values[1]:.2f}')
plt.xlabel("x")
plt.ylabel("Y(x, t)")
plt.title("Advection solution for sinusoidal pulse")
plt.legend()
plt.xticks([])
plt.yticks([])
#plt.axis('off')
plt.axhline(0, color='black', linewidth=1.25)
plt.axvline(0, color='black', linewidth=1.25)
#plt.grid(True)

plt.subplot(1, 3, 3)
u = sinusoidal_pulse_vec(x, t_values[2])
plt.plot(x, u, label=f't = {t_values[2]:.2f}')
plt.xlabel("x")
plt.ylabel("Y(x, t)")
#plt.title("Traveling Sinusoidal Pulse (One Period)")
plt.legend()
plt.xticks([])
plt.yticks([])
#plt.axis('off')
plt.axhline(0, color='black', linewidth=1.25)
plt.axvline(0, color='black', linewidth=1.25)
#plt.grid(True)

# for t in t_values:
#     u = sinusoidal_pulse_vec(x, t)
#     plt.plot(x, u, label=f't = {t:.2f}', color = 'b')

###Plot aesthetics
# plt.xlabel("x (spatial dimension)")
# plt.ylabel("u(x, t)")
# plt.title("Traveling Sinusoidal Pulse (One Period)")
# plt.legend()
# # # plt.grid(True)
# plt.axis('off')
# plt.axhline(0, color='black', linewidth=1.25)
# plt.axvline(0, color='black', linewidth=1.25)


plt.tight_layout()
plt.show()
