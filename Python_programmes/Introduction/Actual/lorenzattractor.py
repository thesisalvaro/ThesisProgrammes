##Lorenz attractor

##Press play

##Alternate between the two plottin lines to obtain the two distinct figures

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

def lorenz(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

t_span = (0, 60)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

y0 = [1, 1, 1]

colors = ['b', 'g', 'r', 'c', 'm']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))] 



## Uncomment the following to obtain the other

# plt.figure(figsize=(7, 3.5))

# for i in tqdm(range(0, 5, 1)):
#     y0[0] = y0[0] + 0.001*i
#     sol = solve_ivp(lorenz, t_span, y0, t_eval=t_eval, method='RK45')
#     plt.plot(sol.t, sol.y[0], label=f"x(0) = {round(y0[0] + 0.001*i, 3)}", color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])

# plt.xlabel("t")
# plt.ylabel("x(t)")
# plt.xlim(0,30)
# plt.title("Lorenz Attractor")
# plt.legend()
# plt.grid()
# plt.show()

##


## Uncomment the following to obtain the other image

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

sol = solve_ivp(lorenz, t_span, [1, 1, 1], t_eval=t_eval, method='RK45')
ax.plot(sol.y[0], sol.y[1], sol.y[2], color='b')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Lorenz attractor")

plt.show()