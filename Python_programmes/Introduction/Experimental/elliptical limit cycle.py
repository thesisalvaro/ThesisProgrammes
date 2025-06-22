##Press play

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System Parameters
a = 1.0       # Frequency parameter
b = 2.0       # Rotation parameter
epsilon = 0.1 # Damping to limit cycle
R = 1.0       # Radius of limit cycle
alpha = 2.0


def limit_cycle(t, z):
    x, y = z
    dxdt = a * x - b * y - epsilon * (x**2 + y**2 - R**2) * x
    dydt = b * x + a * y - epsilon * (x**2 + y**2 - R**2) * y
    return [dxdt, dydt]

def elliptical_limit_cycle(t, z):
    x, y = z
    dxdt = a * x - b * y - epsilon * (x**2 + (y / alpha)**2 - R**2) * x
    dydt = b * x + a * y - epsilon * (x**2 + (y / alpha)**2 - R**2) * y
    return [dxdt, dydt]

# Time span for integration
t_span = (0, 1000)
t_eval = np.linspace(*t_span, 50000)

# Different initial conditions
initial_conditions = [
    (0.1, 0.1),
    (1.5, 0.1),
    (0.1, -1.5),
    (-1.5, -0.1),
    (0.5, 0.8)
]

# Solve the system for each initial condition
solutions = []
for z0 in initial_conditions:
    sol = solve_ivp(elliptical_limit_cycle, t_span, z0, t_eval=t_eval, method="RK45")
    solutions.append(sol)

# Plot the phase portrait
plt.figure(figsize=(8, 6))
for sol, z0 in zip(solutions, initial_conditions):
    plt.plot(sol.y[0], sol.y[1], label=f"IC: {z0}")
    
# Limit cycle
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(R * np.cos(theta), R * np.sin(theta), 'k--', label="Limit Cycle")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Phase Portrait with Rotating Limit Cycle")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()
