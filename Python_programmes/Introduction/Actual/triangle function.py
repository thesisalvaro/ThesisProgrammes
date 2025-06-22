##Triangle function with a discrete delay output

##Press play

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyArrowPatch

def triangle(t, t1, hw):
    x = np.zeros_like(t)
    for i in range(0, len(t)):
        if t[i] < (t1 + hw) and t[i] > (t1 - hw):
            x[i] = 1 - np.abs((t[i] - t1) / hw)
        else:
            x[i] = 0
    return x

t1 = 1
tau = 0.75
hw = 0.5
t = np.linspace(0, 2.5, 10000)

arrow = FancyArrowPatch((0.75, 0.5), (1.5, 0.5),
                         arrowstyle='<->', mutation_scale=15, color='black')



fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t, triangle(t, t1, hw), label=f"X (input)", color='r', linestyle='dashed')
ax.plot(t, triangle(t, t1 + tau, hw), label=f"Y (output)", color='b', linestyle='-.')
ax.add_patch(arrow)
ax.text(1.125, 0.52, 'delay',
        horizontalalignment='center', fontsize=12)

ax.set_xlabel("time")
ax.set_ylabel("Response")
ax.set_ylim(-0.1, 1.5)
ax.set_title("Effects of discrete time delay")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.legend()
ax.grid()

plt.tight_layout()
plt.show()
