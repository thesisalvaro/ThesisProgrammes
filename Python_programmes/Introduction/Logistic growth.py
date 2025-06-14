## Logistic growth
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

t_span = (0, 25)
dt = 0.01
t_steps = int((t_span[-1] - t_span[0]) / dt)
time = np.linspace(t_span[0], t_span[-1], t_steps)


r_values = 1.0
k_values = 5
solx = np.zeros(t_steps)
solxpunto = np.zeros_like(solx)

solx[0] = 7
solxpunto[0] = r_values * solx[0] * (1 - solx[0] / k_values)


for i in tqdm(range(1, t_steps)):
    k1_x = r_values * solx[i-1] * (1 - solx[i-1] / k_values)
 
    aux_x = solx[i-1] + dt * k1_x

    k2_x = r_values * aux_x * (1 - aux_x / k_values)
    
    solx[i] = solx[i-1] + 0.5 * dt * (k1_x + k2_x)

    solxpunto[i] = r_values * solx[i] * (1 - solx[i] / k_values)

print(solx)
print(solxpunto)


# Plot the results
plt.figure(figsize=(8, 4))

# Plot x(t)

# for i in range(0, long):
#     plt.plot(time, solx[i], label=f'μ = {mu_values[i % len(mu_values)]}', color=colors[i % len(colors)])

# for i in range(0, long):
#     plt.plot(time, solx[i], label=f'(x0, y0) = ({xin[i % len(xin)]}, {yin[i % len(yin)]})', color=colors[i % len(colors)])

# # plt.subplot(2, 1, 1)
# # plt.plot(time, x, label=r"$x(t)$", color='blue')
# plt.xlabel("Time")
# plt.ylabel(r"$x(t)$")
# plt.title(r"Van der Pol Oscillator: $x(t)$")
# plt.legend()
# plt.grid()

# Phase space plot (y vs. x)
x1 = np.linspace(-5, 5, 10000)
x2 = np.zeros(len(x1))

# plt.subplot(2, 1, 2)

x = np.linspace(0, 7, t_steps)
xx = np.linspace(-0.2, 7, t_steps)
y = np.linspace(-1.5, 1.5, t_steps)
xprime = np.zeros_like(y)
y0 = np.zeros_like(x)
xpunto = r_values * x * (1 - x / k_values)

arrow_positions = [2.2, 6.3]
arrow_directions = [0.3, -0.3]

# for i in range(0, long):
#     plt.plot(solx[i], soly[i], label=f'μ = {mu_values[i % len(mu_values)]}', color=colors[i % len(colors)])

dot = '\u0307'

plt.plot(x, xpunto, color='b')

for pos, direction in zip(arrow_positions, arrow_directions):
    plt.arrow(pos, 0, direction, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')

#axis en negrita
plt.plot(xx, y0, 'k')
plt.plot(xprime, y, 'k')

#puntos 
plt.plot(0, 0, 'ko', fillstyle='none')
plt.plot(5, 0, 'ko')


plt.xlim(-0.2, 7)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('x' + dot)
plt.title("Logistic growth")
plt.grid()

plt.tight_layout()
plt.show()