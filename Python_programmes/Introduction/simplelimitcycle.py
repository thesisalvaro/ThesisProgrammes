##simple limit cycle with Hopf?
##Vanderpol osc 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Time span
t_span = (0, 25)  # Time range for simulation
dt = 0.01         # Time step for numerical evaluation
t_steps = int((t_span[-1] - t_span[0]) / dt)
time = np.linspace(t_span[0], t_span[-1], t_steps)

# Initial conditions
r = np.zeros(t_steps)
theta = np.zeros(t_steps)

xin = 0.01
yin = 1
#x[0] = xin[]  # Initial x value
#y[0] = 3.0  # Initial y value

mu_values = np.linspace(1, 5, 5)
rfp_valuespos = np.sqrt(mu_values)
rfp_valuesneg = -np.sqrt(mu_values)
long = len(mu_values)
solr = np.zeros((long, t_steps))
soltheta = np.zeros_like(solr)
solrdot = np.zeros_like(solr)

omega = 1
b = 20
##Integration using Runge-Kutta 2nd order method
for j in range(0, long):
    mu = mu_values[j]
    r[0] = xin
    theta[0] = yin
    for i in tqdm(range(1, t_steps)):
        k1_r = r[i-1] * (mu - r[i-1] **2)
        k1_theta = omega + b * r[i-1] **2
    
        aux_r = r[i-1] + dt * k1_r
        aux_theta = theta[i-1] + dt * k1_theta
    
        k2_r = aux_r * (mu - aux_r **2)
        k2_theta = omega + b * aux_r **2
    
        r[i] = r[i-1] + 0.5 * dt * (k1_r + k2_r)
        theta[i] = theta[i-1] + 0.5 * dt * (k1_theta + k2_theta)
    
    rdot = r * (mu - r **2)
    solr[j, :] = r
    soltheta[j, :] = theta
    solrdot[j, :] = rdot
    

# Plot the results
plt.figure(figsize=(12, 6))

colors = ['b', 'b', 'b', 'b', 'b']

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


# plt.subplot(2, 1, 2)


# for i in range(0, long):
#     plt.plot(solx[i], soly[i], label=f'μ = {mu_values[i % len(mu_values)]}', color=colors[i % len(colors)])

for i in range(0, long):
    plt.plot(solr[i], solrdot[i], label=f'μ = {mu_values[i % len(mu_values)]}', color=colors[i % len(colors)])

##el importante es este de arriba!!! 

# plt.plot(mu_values, rfp_valuespos, label=f'μ^1/2', color = 'b')
# plt.plot(mu_values, rfp_valuesneg, label=f'-μ^1/2', color = 'b', linestyle = 'dashed')

# arrow_spacing = 200  # Number of points to skip between arrows
# for i in range(len(xin)-1):
#     for j in range(0, len(solx[i]) - 1, arrow_spacing):
#         dx = solx[i][j + 1] - solx[i][j]
#         dy = soly[i][j + 1] - soly[i][j]
#         plt.annotate('', xy=(solx[i][j + 1], soly[i][j + 1]), xytext=(solx[i][j], soly[i][j]),
#                      arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)], lw=1.5))


dot = '\u0307'

x1 = np.linspace(-5, 5, 10000)
x2 = np.zeros(len(x1))

plt.plot(x1, x2, color='black')
plt.plot(x2, x1, color='black')
plt.xlim(-.1, 2)
plt.ylim(-1.5, 5)
plt.xlabel(r'r')
plt.ylabel(r'r' + dot)
plt.title("Phase Space Plot")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()