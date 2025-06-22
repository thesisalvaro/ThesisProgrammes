##advection in an open loop

##Press play

import numpy as np
import matplotlib.pyplot as plt

a = 1.0
omega = 0.5 * np.pi
T = (2*np.pi) / omega
x_min, x_max = 0, 10
Nx = 500
t_values = [2, 5, 8]
x = np.linspace(x_min, x_max, Nx)

def sinusoidal_pulse(x, t):
    tau = t - x / a
    if 0 <= tau <= T / 2:
        return np.sin(omega * tau)
    else:
        return 0.0

sinusoidal_pulse_vec = np.vectorize(sinusoidal_pulse, excluded=['t'])

plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
u = sinusoidal_pulse_vec(x, t_values[0])
plt.plot(x, u, label=f't = {t_values[0]:.2f}')
plt.xlabel("x")
plt.ylabel("Y(x, t)")

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
plt.legend()
plt.xticks([])
plt.yticks([])
#plt.axis('off')
plt.axhline(0, color='black', linewidth=1.25)
plt.axvline(0, color='black', linewidth=1.25)
#plt.grid(True)

plt.tight_layout()
plt.show()
