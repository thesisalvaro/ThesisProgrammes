##DiffPDE Runge Kutta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm

def X_equation(x, y_delta, k1, K, m, q1):
    return k1/(np.power(K,m)+np.power(y_delta,m))-q1*x


def Y_equation(t, x, y, k2, q2, D, dx, mu): 
    y[1:-1] = (D/(dx**2))*(-2*y[1:-1]+y[:-2]+y[2:]) - mu * y[1:-1]
    y[0] = k2*x-q2*y[0]
    return y

# k1 = 0.1
# K = 0.8
# m = 12.0
# q1 = 0.03
# q2 = 0.03
# k2 = 0.01
# D = 0.03
# mu = 0.01
# Delta = 5


##advection parameters
k1 = 0.1
K = 1.0
m = 6.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
D = 0.06                     # Advection speed
mu = 0.0
Delta = 7.5



a_values = np.linspace(0.4, 0.425, 6)
m_values = [4.8, 4.9]
q2_values = [0.015, 0.02, 0.025, 0.03, 0.13, 0.135, 0.14, 0.145]
k1_values = [0.085, 0.09, 0.095, 0.1]
k2_values = [0.08, 0.085, 0.09, 0.095, 0.1]
K_values = [0.8]
Delta_values = [0.5, 1, 2, 5, 7.5, 10, 15, 20]
mu_values = [0.0]
D_values = [0.1]



# Grid
C = 10.0                   # Spatial domain limit
Nx = 501                   # Number of spatial points
x = np.linspace(0, C, Nx)  # Spatial grid
dx = x[1] - x[0]            # Spatial step

delta_idx = np.abs(x-Delta).argmin()

# Time span
t_span = (0, 1000)           # Time range for simulation
dt = 0.05                      # Time step for numerical evaluation

cfl_limit = (dx**2) / (2 * D)
if dt > cfl_limit:
    print(f"(dt = {dt}) > CFL limit ({cfl_limit:.5f}). Set dt to {cfl_limit:.5f}.")
    dt = cfl_limit

t_steps = int(t_span[-1]/dt) 


# Initial conditions
X = np.ones([t_steps, len(mu_values)])     # Initial value for X
Y = np.zeros([Nx, t_steps]) # Initial values for Y(x, t)
#Y0 = np.zeros([t_steps])
#Y0[0] = 0

time = np.linspace(0,t_span[-1],t_steps)   

#plt.figure(figsize=(12, 6))
 

# cfl_limit = (dx**2) / (2 * D_values)
# if dt > cfl_limit:
#     print(f"Warning: Time step (dt = {dt}) exceeds CFL limit ({cfl_limit:.5f}). Adjusting dt to {cfl_limit:.5f}.")
#     dt = cfl_limit

#def initial_condition(t):
 #   return np.maximum(0, -np.sin(np.pi*t))

plt.figure(figsize=(12, 6))
 


for j in tqdm(range(0, len(mu_values))):
    mu = mu_values[j]
    for i in tqdm(range(1,t_steps)):
        k1_x = X_equation(X[i-1, j], Y[delta_idx,i-1], k1, K, m, q1)
        #k1_y0 = y0_equation(X[i-1], Y0[i-1], k2, q2)
        k1_y = Y_equation((i-1)*dt, X[i-1, j], Y[:,i-1].copy(), k2, q2, D, dx, mu)
        

        aux_x = X[i-1, j]+dt*k1_x
        #aux_y0 = Y0[i-1] + dt*k1_y0
        aux_y = Y[:,i-1]+dt*k1_y
        
        
        k2_x = X_equation(aux_x, aux_y[delta_idx], k1, K, m, q1)
        #k2_y0 = y0_equation(aux_x, aux_y0, k2, q2)
        k2_y = Y_equation(i*dt, aux_x, aux_y, k2, q2, D, dx, mu)

        
        X[i, j] = X[i-1, j]+0.5*dt*(k1_x+k2_x)
        #Y0[i] = Y0[i-1] + 0.5*dt*(k1_y0+k2_y0)
        Y[:,i] = Y[:,i-1]+0.5*dt*(k1_y+k2_y)

    



    # plt.subplot(2, 1, j + 1)
    # plt.imshow(Y[::-1,:], cmap="viridis", aspect="auto", extent=[0,t_span[-1],0,100])
    # plt.xlim(0, 1000)
    # plt.colorbar()
    # plt.xlabel("t")
    # plt.ylim(0, 7)
    # plt.ylabel("x")
    # plt.title(f'Y(x, t) with μ = {mu_values[j]}')

    # plt.subplot(2, 1, 2)
    # plt.imshow(Y[::-1,:], cmap="viridis", aspect="auto", extent=[0,t_span[-1],0,100])
    # plt.xlim(0, 1000)
    # plt.colorbar()
    # plt.xlabel("t")
    # plt.ylim(0, 20)
    # plt.ylabel("x")
    # plt.title("Y(x, t)")

    # # # plt.subplot(3, 1, 1)
    # # # plt.plot(time, X[:, j], label=f'μ ={mu_values[j]}')
    # # # plt.xlabel("t")
    # # # plt.ylabel("x(t)")
    # # # plt.title("x(t) solution")
    # # # plt.legend()
    # # # plt.grid()

    # # # plt.subplot(3, 1, 2)
    # # # plt.plot(time, Y[0, :], label=f'μ ={mu_values[j]}')
    # # # plt.xlabel("t")
    # # # plt.ylabel("Y(0,t)")
    # # # plt.title("Y(0,t) solution")
    # # # plt.legend()
    # # # plt.grid()

    # # # plt.subplot(3, 1, 3)
    # # # plt.plot(time, Y[delta_idx, :], label=f'μ ={mu_values[j]}')
    # # # plt.xlabel("t")
    # # # plt.ylabel("Y(Delta, t)")
    # # # plt.title("Y(Delta, t) solution")
    # # # plt.legend()
    # # # plt.grid()
    # plt.plot(time, X[:, j], label="X(t)", color='blue')

 
 
# Plot the results
# plt.figure(figsize=(12, 6))
 
# Plot X(t)
# plt.subplot(3, 1, 1)

downsample_factor = 100  # Adjust as needed
Y_downsampled = Y[:, ::downsample_factor]

# colors = ['b', 'g', 'y', 'r', 'k']

# for j in tqdm(range(0, len(mu_values))):
#     plt.plot(time, X[:, j], label=f'μ ={mu_values[j]}')
    
plt.subplot(2, 1, 1)
plt.plot(time, X, label="X(t)", color='blue')
plt.plot(time, Y[-10, :], label="Y(10,t)", color='r')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("x(t) solution")
plt.legend()
plt.grid()
 

print('shape of Y is', np.shape(Y))
##Plot Y(x, t) for selected x values
plt.subplot(2, 1, 2)
## plt.imshow(Y[::-1,:], cmap="viridis", vmax = 5, aspect="auto", extent=[0,t_span[-1],0,100]) ## aqui está metio el vmax
plt.imshow(Y_downsampled[::-1, :], cmap="viridis", aspect="auto", extent=[0,t_span[-1],0,100])
plt.ylim(0, 10)
# plt.xlim(0, 1000)
plt.colorbar()
plt.xlabel("t")
# plt.ylim(0, 20)
plt.ylabel("x")
plt.title("Y(x, t)")

 
plt.tight_layout()
plt.show()