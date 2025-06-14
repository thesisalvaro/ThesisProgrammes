##stair function
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
# ax.set_xlim(0.3, 2.5)
ax.set_ylim(-0.1, 1.5)
ax.set_title("Effects of discrete time delay")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.legend()
ax.grid()

plt.tight_layout()
plt.show()

# # Parameters for the triangles
# t1 = 2        # Center of the first triangle
# tau = 1       # Half-width of the triangle
# delay = 3     # Time delay between the triangles

# # Generate the first triangle
# x1 = np.linspace(t1 - tau, t1 + tau, 100)
# y1 = 1 - np.abs((x1 - t1) / tau)  # Triangle function

# # Generate the second triangle
# x2 = np.linspace(t1 + delay - tau, t1 + delay + tau, 100)
# y2 = 1 - np.abs((x2 - (t1 + delay)) / tau)

# # Create the plot
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(x1, y1, label='Triangle 1', color='blue')
# ax.plot(x2, y2, label='Triangle 2', color='orange')

# # Annotate the delay with an arrow
# arrow = FancyArrowPatch((t1 + tau, 0.5), (t1 + delay - tau, 0.5),
#                          arrowstyle='|-|', mutation_scale=15, color='black')
# ax.add_patch(arrow)
# ax.text((t1 + tau + t1 + delay - tau) / 2, 0.6, 'delay',
#         horizontalalignment='center', fontsize=12)

# # Add labels and legend
# ax.set_xlabel('Time', fontsize=14)
# ax.set_ylabel('Amplitude', fontsize=14)
# ax.legend(fontsize=12)
# ax.grid(True, linestyle='--', alpha=0.6)
# ax.set_title('Triangle Functions with Delay', fontsize=16)

# # Show the plot
# plt.tight_layout()
# plt.show()