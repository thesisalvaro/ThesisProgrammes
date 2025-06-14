# ##prueba dif

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.integrate as integrate
from tqdm import tqdm

matplotlib.use('QtAgg')  # Use QtAgg backend for interactive plots

# --- Model equations ---
def X_equation(x, y_delta, m, k1, K, q1):
    return k1 / (K**m + y_delta**m) - q1 * x

# def Y_equation(x, y, q2, k2):
#     return k2 * x - q2 * y

z = np.zeros(499)

def Y_equation(t, y, D, dx, mu): 
#   y[1:-1] = (D/(dx**2))*(-2*y[1:-1]+y[:-2]+y[2:]) - mu * y[1:-1]
    y[1:-1] = (D/(dx**2))*(-2*y[1:-1]+y[:-2]+y[2:] - mu * y[1:-1])
    y[0] = 1
    y[-1] = 0
    
#    y[0] = np.abs(np.sin(np.pi/200 * t))
    return y

def kernel(tau, D, mu, Delta):
    return (1 / (np.pi * D * tau)**0.5) * np.exp(-Delta**2 / (4 * D * tau) - mu * tau)


def integrand(t, x, D, mu):
    return (x) /((4*np.pi * D *(t)**3)**0.5) * np.exp(-x**2 / (4 * D * (t)) - mu * (t))

def ker2(t, x, D, mu):
    return integrate.quad(integrand, 0, t, args=(x, D, mu, t))

span = [0, 5]
sx = 101
space = np.linspace(span[0], span[1], sx)
ddx = space[1] - space[0]



# --- Parameters ---
k1 = 0.1
K = 1.0
m = 6.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
D = 0.06
mu = 0.0
Delta = 7.5

# --- Time settings ---
t_span = (0, 100)
dt = 0.01
t_steps = int(t_span[-1] / dt)
time = np.linspace(*t_span, t_steps)

C = 5.0                   # Spatial domain limit
Nx = 501                   # Number of spatial points
x = np.linspace(0, C, Nx)  # Spatial grid
dx = x[1] - x[0]      

delta_idx = np.abs(x-Delta).argmin()

cfl_limit = (dx**2) / (2 * D)
if dt > cfl_limit:
    print(f"(dt = {dt}) > CFL limit ({cfl_limit:.5f}). Set dt to {cfl_limit:.5f}.")
    dt = cfl_limit

# --- Initialize arrays ---
Y = np.zeros((Nx, t_steps))

sol = np.zeros([sx, t_steps])


# --- Time integration using RK2 ---
Y[0, 0] = 1

for i in tqdm(range(1, t_steps)):

    Y[:, 0] = 0
    Y[0, 0] = 1

    k1_y = Y_equation(i, Y[:, i-1], D, dx, mu)

    aux_y = Y[:, i-1] + dt * k1_y

    k2_y = Y_equation(i, aux_y, D, dx, mu)

    Y[:, i] = Y[:, i-1] + 0.5 * dt * (k1_y + k2_y)

# sol[0, 0] = 1
# sol[1:, 0] = 0

# for j in tqdm(range(0, len(space))):
#     for i in tqdm(range(1, t_steps)):
#         xx = space[j]
#         sol[j, i] = integrate.quad(integrand, 0, i*dt, args=(xx, D, mu, i*dt))[0]



# --- Plot results ---
plt.figure(figsize=(8, 4))

for t_idx in range(0, t_steps, t_steps // 5):
    plt.plot(x, kernel(time[t_idx], D, mu, x), label=f't = {time[t_idx]:.2f}', linestyle='dashed')
#     plt.plot(space, sol[:, t_idx], label=f't = {time[t_idx]:.2f}', linestyle='dashed')


# plt.plot(time, [200, :], label="Ydelta(t), μ = 0", linestyle='dashed')


plt.xlabel("x")
plt.ylabel("Y(x,t)")
# plt.xlim(-0.25, 4)
plt.title("Diffusion equation for a single pulse")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

###


# plt.subplot(3, 1, 1)
# plt.plot(time, X[:, 0], label="X(t), μ = 0", color='b')
# plt.xlabel("t")
# plt.ylabel("X")
# plt.title("Time series of X(t)")
# plt.legend()
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.plot(time, Yprime[:, 0], label="Ydelta(t), μ = 0", color='r', linestyle='dashed')
# plt.xlabel("t")
# plt.ylabel("Y")
# plt.title("Time series of Ydelta(t)")
# plt.legend()
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.plot(time, Y[:, 0], label="Y(0,t), μ = 0", color='y')
# plt.xlabel("t")
# plt.ylabel("Y")
# plt.title("Time series of Y(0,t)")
# plt.legend()
# plt.grid()


# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # --- Parameters ---
# D = 0.06                # Diffusion coefficient
# mu = 0.0               # Decay rate
# C = 10.0               # Spatial domain length
# Nx = 501               # Number of spatial points
# dx = C / (Nx - 1)      # Spatial step size
# x = np.linspace(0, C, Nx)  # Spatial grid

# t_span = (0, 0.01)       # Time span
# dt = 0.001              # Time step
# t_steps = int(t_span[-1] / dt)  # Number of time steps
# time = np.linspace(*t_span, t_steps)

# # Ensure CFL condition
# cfl_limit = (dx**2) / (2 * D)
# if dt > cfl_limit:
#     print(f"(dt = {dt}) > CFL limit ({cfl_limit:.5f}). Adjusting dt to {cfl_limit:.5f}.")
#     dt = cfl_limit

# # --- Initialize arrays ---
# Y = np.zeros((Nx, t_steps))  # Concentration array
# x_prime_idx = Nx // 2        # Initial concentration at x'
# Y[x_prime_idx, 0] = 1.0      # Initial condition: Y(x', 0) = 1

# # --- Function for computing RHS ---
# def Y_equation(t, y, D, dx, mu): 
#     laplacian = (D / (dx**2)) * (-2 * y[1:-1] + y[:-2] + y[2:])
#     decay = -mu * y[1:-1]
#     result = laplacian + decay
#     y[1:-1] = result
#     y[0] = 0     # Dirichlet boundary condition at x=0
#     y[-1] = 0    # Dirichlet boundary condition at x=C
#     return y

# # --- Time integration using RK2 ---
# for i in tqdm(range(1, t_steps)):
#     k1_y = Y_equation((i-1)*dt, Y[:, i-1], D, dx, mu)
#     aux_y = Y[:, i-1] + dt * k1_y
#     k2_y = Y_equation(i*dt, aux_y, D, dx, mu)
#     Y[:, i] = Y[:, i-1] + 0.5 * dt * (k1_y + k2_y)
#     Y[:, i] = np.maximum(Y[:, i], 0)

# # --- Visualization ---
# plt.figure(figsize=(10, 6))
# for t_idx in range(0, t_steps, t_steps // 10):  # Plot 10 time slices
#     plt.plot(x, Y[:, t_idx], label=f't = {time[t_idx]:.2f}')
# plt.xlabel('x')
# plt.ylabel('Y(x, t)')
# plt.title('Diffusion Equation Solution')
# plt.legend()
# plt.grid()
# plt.show()


