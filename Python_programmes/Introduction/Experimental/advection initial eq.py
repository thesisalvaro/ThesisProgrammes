import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


q = 0.03
Delta = 7.5
m = 5
K = 2.1147
k1 = 2
k2 = 2
k3 = (K ** (m + 1)) / (k1 * k2)
a = 0.405
tau = q * Delta / a


x_max = 200
nx = 1000
x = np.linspace(0, x_max, nx)
dx = x[1] - x[0]

t_span = (0, 200)
t_eval = np.linspace(*t_span, 1000)


X0 = 1
Y0 = np.zeros(nx)

def R_X(X):
    return X

initial_conditions = np.concatenate([[X0], Y0])

def system(t, u):
    X = u[0]
    Y = u[1:]

    Y_N = Y[-1]

    dXdt = k1 / (K**m + Y_N**m) - q * X

    dYdt = np.zeros_like(Y)
    dYdt[1:-1] = k2 * X - q * Y[1:-1] - a * (Y[1:-1] - Y[:-2]) / dx

    dYdt[0] = R_X(X)
    #dYdt[-1] = k2 * X - q * Y[-1] - a * (Y[-1] - Y[-2]) / dx
    #dYdt[-1] = - q * Y[-1]  # At x = Delta, Y_N(t)

    print('dimension', np.ndim(dYdt),', size', np.size(dYdt), ', shape', np.shape(dYdt))

    return np.concatenate([[dXdt], dYdt])


solution = solve_ivp(
    system, t_span, initial_conditions, method='RK45', t_eval=t_eval
)

X_sol = solution.y[0, :]
Y_sol = solution.y[1:, :]

plt.figure(figsize=(10, 6))
plt.plot(t_eval, X_sol, label='X(t)', color='blue')
plt.title('Concentration of X(t) over Time')
plt.xlabel('Time t')
plt.ylabel('Concentration X(t)')
plt.grid(True)
plt.legend()
plt.show()

#x_indices = [0, nx // 4, nx // 2, 3 * nx // 4, -1]
x_indices = [0, 75]
plt.figure(figsize=(10, 6))
for idx in x_indices:
    plt.plot(t_eval, Y_sol[idx, :], label=f'Y(x={x[idx]:.2f}, t)')
plt.title('Concentration of Y(x, t) over Time for Different x')
plt.xlabel('Time t')
plt.ylabel('Concentration Y(x, t)')
plt.grid(True)
plt.legend()
plt.show()


#t_indices = [0, len(t_eval) // 4, len(t_eval) // 2, 3 * len(t_eval) // 4, -1]  # Select distinct time points
t_indices = [0, 1, 2, 3, 4, 5, 50, 100, 250, 500]
plt.figure(figsize=(10, 6))
for idx in t_indices:
    plt.plot(x, Y_sol[:, idx], label=f'Y(x, t={t_eval[idx]:.2f})')
plt.title('Concentration of Y(x, t) over Space for Different t')
plt.xlabel('Position x')
plt.ylabel('Concentration Y(x, t)')
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(12, 8))
T, X = np.meshgrid(t_eval, x)
plt.pcolormesh(T, X, Y_sol, shading='auto', cmap='viridis')
cbar = plt.colorbar()
cbar.set_label('Concentration Y(x, t)')
plt.title('Phase Space Heatmap of Y(x, t)')
plt.xlabel('Time t')
plt.ylabel('Position x')
plt.show()
