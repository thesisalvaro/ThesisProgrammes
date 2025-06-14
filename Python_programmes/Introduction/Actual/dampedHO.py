##Damped Harmonic Oscillator
##Press play to obtain one figure, follow the comments to obtain the other

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

t_span = (0, 20)            ##tspan goes to 20 (stable FP) or 4.5 (unstable FP)
dt = 0.01
t_steps = int((t_span[-1] - t_span[0]) / dt)
time = np.linspace(t_span[0], t_span[-1], t_steps)

x = np.zeros(t_steps)
y = np.zeros(t_steps)

xin = [-1, -0.5, 0.5, 1]
yin = [-1, 1, -1, 1]


gamma_values = [0.4]            ##Change this to -0.4 or 0.4 to obtain one or the other
k_values = 4
m_values = 1
long = len(xin)
solx = np.zeros((long, t_steps))
soly = np.zeros_like(solx)


for j in range(0, long):
    gamma = gamma_values[0]
    x[0] = xin[j]
    y[0] = yin[j]
    for i in tqdm(range(1, t_steps)):
        k1_x = y[i-1]
        k1_y = - (gamma / m_values) * y[i-1] - (k_values / m_values) * x[i-1]
    
        aux_x = x[i-1] + dt * k1_x
        aux_y = y[i-1] + dt * k1_y
    
        k2_x = aux_y
        k2_y = -(gamma / m_values) * aux_y - (k_values / m_values) * aux_x
    
        x[i] = x[i-1] + 0.5 * dt * (k1_x + k2_x)
        y[i] = y[i-1] + 0.5 * dt * (k1_y + k2_y)
    solx[j, :] = x
    soly[j, :] = y



plt.figure(figsize=(7, 3.5))

colors = ['b', 'g', 'r', 'y', 'k']

x1 = np.linspace(-5, 50, 10000)
x2 = np.zeros(len(x1))



for i in range(0, long):
    plt.plot(solx[i], soly[i], label=f'(x0, y0) = ({xin[i % len(xin)]}, {yin[i % len(yin)]})', color=colors[i % len(colors)])


arrow_spacing = 100 
for i in range(len(xin)):
    for j in range(0, len(solx[i])- 1300, arrow_spacing):       ##change len(solx[i])- 1300 to len(solx[i]) only or viceversa
        dx = solx[i][j + 1] - solx[i][j]
        dy = soly[i][j + 1] - soly[i][j]
        plt.annotate('', xy=(solx[i][j + 1], soly[i][j + 1]), xytext=(solx[i][j], soly[i][j]),
                     arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)], lw=1.5))




dot = '\u0307'
plt.plot(x1, x2, color='black')
plt.plot(x2, x1, color='black')
# plt.xlim(-3, 3)
# plt.ylim(-5, 5)           ##Comment/uncomment these
plt.xlim(-1.2, 1.2)         ##Comment/uncomment these
plt.ylim(-2.05, 2.05)
plt.xlabel('x')
plt.ylabel('x' + dot)
plt.title("Damped harmonic oscillator")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()