import numpy as np
import matplotlib.pyplot as plt
 
import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm

def X_equation(x, y_delta, k1, K, m, q1):
    return k1/(np.power(K,m)+np.power(y_delta,m))-q1*x

# def Y_equation(t, x, y, k2, q2, a, dx): 
#     y[1:] =k2*x-q2*y[1:]-(a/dx)*(y[1:]-y[:-1]) ## y[:-1] - y[1:] pal otro lao
#     y[-1] = 0
#     return y

# def Y_equation(t, y, a, dx):
#     yn = np.zeros_like(y)
#     yn[1:] = y[1:] - (a/dx)*(y[1:]-y[:-1])
#     yn[0] = y[0]
#     yn[-1] = y[-1]
#     return yn

def Y_equation(t, x, y, k2, q2, a, dx): 
    y[1:] = -(a/dx)*(y[1:]-y[:-1])
    y[0] = k2*x-q2*y[0]
    return y

#def y0_equation(x, y0, k2, q2):
 #   return k2 * x - q2 * y0

# # Parameters
# k1 = 0.1
# K = 1
# m = 5.0
# q1 = 0.03
# q2 = 0.03
# k2 = 0.1
# a = 0.405                     # Advection speed

# Delta = 7.5                 # Position for Y(x = Delta)


# # Grid
# C = 500.0                   # Spatial domain limit
# Nx = 1001                   # Number of spatial points
# x = np.linspace(0, Delta, Nx)  # Spatial grid
# dx = x[1] - x[0]            # Spatial step
# delta_idx = np.abs(x-Delta).argmin()

k1 = 0.1
K = 1.0
m = 5.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
a = 0.35                     # Advection speed
Delta = 7.5

# Grid
C = 10.0                   # Spatial domain limit
Nx = 1001                   # Number of spatial points
x = np.linspace(0, C, Nx)  # Spatial grid
dx = x[1] - x[0]            # Spatial step

                 # Position for Y(x = Delta)
delta_idx = np.abs(x-Delta).argmin()

# Time span
t_span = (0, 5000)           # Time range for simulation
dt = 0.05                   # Time step for numerical evaluation
t_steps = int(t_span[-1]/dt)

# Initial conditions
X = np.ones([t_steps])     # Initial value for X
Y = np.zeros([Nx,t_steps]) # Initial values for Y(x, t)
#Y0 = np.zeros([t_steps])
#Y0[0] = 0

#def initial_condition(t):
 #   return np.maximum(0, -np.sin(np.pi*t))

for i in tqdm(range(1,t_steps)):
    k1_x = X_equation(X[i-1], Y[delta_idx,i-1], k1, K, m, q1)
    #k1_y0 = y0_equation(X[i-1], Y0[i-1], k2, q2)
    k1_y = Y_equation((i-1)*dt, X[i-1], Y[:,i-1].copy(), k2, q2, a, dx)
    

    aux_x = X[i-1]+dt*k1_x
    #aux_y0 = Y0[i-1] + dt*k1_y0
    aux_y = Y[:,i-1]+dt*k1_y
    
    
    k2_x = X_equation(aux_x, aux_y[delta_idx], k1, K, m, q1)
    #k2_y0 = y0_equation(aux_x, aux_y0, k2, q2)
    k2_y = Y_equation(i*dt, aux_x, aux_y, k2, q2, a, dx)
    
    X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
    #Y0[i] = Y0[i-1] + 0.5*dt*(k1_y0+k2_y0)
    Y[:,i] = Y[:,i-1]+0.5*dt*(k1_y+k2_y)

    
    #Y[0,i] = Y0[i]
    #Y[-1,i] = 0
 
time = np.linspace(0,t_span[-1],t_steps)   
 
# Plot the results
plt.figure(figsize=(12, 6))
 
# Plot X(t)
plt.subplot(2, 1, 1)
plt.plot(time, X, label="X(t)", color='blue')
plt.xlabel("Time")
plt.title("Solution for X(t)")
plt.legend()
plt.grid()
 
# Plot Y(x, t) for selected x values
plt.subplot(2, 1, 2)
plt.imshow(Y[::-1,:], cmap="viridis", aspect="auto", extent=[0,t_span[-1],0,Nx])
plt.colorbar()
plt.xlabel("Time")
plt.ylabel("Y(x, t)")
plt.title("Solution for Y(x, t)")

 
plt.tight_layout()
plt.show()