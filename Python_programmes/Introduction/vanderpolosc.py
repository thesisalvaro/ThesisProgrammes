##Vanderpol osc 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Van der Pol oscillator parameters
  # Nonlinearity and damping parameter

# Time span
t_span = (0, 25)  # Time range for simulation
dt = 0.01         # Time step for numerical evaluation
t_steps = int((t_span[-1] - t_span[0]) / dt)
time = np.linspace(t_span[0], t_span[-1], t_steps)

# Initial conditions
x = np.zeros(t_steps)
y = np.zeros(t_steps)

xin = [2.0, -2.0, 1.0, -1.0]
yin = [0, 3.0, 0.0, 0.0]
#x[0] = xin[]  # Initial x value
#y[0] = 3.0  # Initial y value

mu_values = [0.5, 1, 1.5, 2]
long = len(xin)
solx = np.zeros((long, t_steps))
soly = np.zeros_like(solx)

# Integration using Runge-Kutta 2nd order method
for j in range(0, long):
    mu = mu_values[j]
    x[0] = xin[0]
    y[0] = yin[0]
    for i in tqdm(range(1, t_steps)):
        k1_x = y[i-1]
        k1_y = mu * (1 - x[i-1]**2) * y[i-1] - x[i-1]
    
        aux_x = x[i-1] + dt * k1_x
        aux_y = y[i-1] + dt * k1_y
    
        k2_x = aux_y
        k2_y = mu * (1 - aux_x**2) * aux_y - aux_x
    
        x[i] = x[i-1] + 0.5 * dt * (k1_x + k2_x)
        y[i] = y[i-1] + 0.5 * dt * (k1_y + k2_y)
    solx[j, :] = x
    soly[j, :] = y


# Plot the results
plt.figure(figsize=(8, 4))

colors = ['b', 'g', 'y', 'r', 'k']

# Plot x(t)

for i in range(0, long):
    plt.plot(time, solx[i], label=f'μ = {mu_values[i % len(mu_values)]}', color=colors[i % len(colors)])

# for i in range(0, long):
#     plt.plot(time, solx[i], label=f'(x0, y0) = ({xin[i % len(xin)]}, {yin[i % len(yin)]})', color=colors[i % len(colors)])

# plt.subplot(2, 1, 1)
# plt.plot(time, x, label=r"$x(t)$", color='blue')
plt.xlabel("Time")
plt.ylabel(r"x(t)")
plt.title(r"Van der Pol Oscillator")
plt.legend()
plt.grid()

# Phase space plot (y vs. x)
x1 = np.linspace(-5, 5, 10000)
x2 = np.zeros(len(x1))

# plt.subplot(2, 1, 2)


# for i in range(0, long):
#     plt.plot(solx[i], soly[i], label=f'μ = {mu_values[i % len(mu_values)]}', color=colors[i % len(colors)])

# # # for i in range(0, long-1):
# # #     plt.plot(solx[i], soly[i], label=f'(x0, y0) = ({xin[i % len(xin)]}, {yin[i % len(yin)]})', color=colors[i % len(colors)])

# # # plt.plot(solx[4], soly[4], color=colors[4])

# # # arrow_spacing = 200  # Number of points to skip between arrows
# # # for i in range(len(xin)-1):
# # #     for j in range(0, len(solx[i]) - 1, arrow_spacing):
# # #         dx = solx[i][j + 1] - solx[i][j]
# # #         dy = soly[i][j + 1] - soly[i][j]
# # #         plt.annotate('', xy=(solx[i][j + 1], soly[i][j + 1]), xytext=(solx[i][j], soly[i][j]),
# # #                      arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)], lw=1.5))


# # # plt.plot(x1, x2, color='black')
# # # plt.plot(x2, x1, color='black')
# # # plt.xlim(-3, 3)
# # # plt.ylim(-4, 4)
# # # plt.xlabel(r"$x$")
# # # plt.ylabel(r"$y$")
# # # plt.title("Phase Space Plot")
# # # plt.legend()
# # # plt.grid()

plt.tight_layout()
plt.show()