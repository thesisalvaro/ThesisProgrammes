import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

# Fixed parameters
K_s = 1.4
b_deg = 0.1
K = 30

# Function definitions
def hill_function(C, m):
    return C**m / (K**m + C**m)

def model(Y, t, params):
    """
    Delayed differential equation system.
    """
    C = Y(t)[0]
    A = Y(t)[1]
    C_lag = Y(t - params['tau'])[0] if t - params['tau'] > 0 else Y(0)[0]

    m = params['m']
    dCdt = K_s - b_deg * C * hill_function(C_lag, m)
    dAdt = hill_function(C, m) - A
    return np.array([dCdt, dAdt])

# Parameters for the three cases
cases = [
    {'tau': 10, 'm': 10},  # tau=10, m=10
    {'tau': 10, 'm': 5},   # tau=10, m=5
    {'tau': 5, 'm': 2}     # tau=5, m=2
]

# Initial conditions and time span
t_span = (0, 200)
t_eval = np.linspace(*t_span, 1000)
C0 = 1.0
A0 = hill_function(C0, cases[0]['m'])  # Steady-state approximation
y0 = lambda t: np.array([C0, A0])  # Constant history function

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figure size for better aspect ratio
for i, case in enumerate(cases):
    sol = ddeint(model, y0, t_eval, fargs=(case,))

    ax_left = axs[i]
    ax_right = ax_left.twinx()  # Create a twin y-axis for A(t)

    ax_left.plot(t_eval, sol[:, 0], label='C(t)', color='blue')  # C(t)
    ax_right.plot(t_eval, sol[:, 1], label='A(t)', color='red')   # A(t)

    ax_left.set_title(f"Tau = {case['tau']}, m = {case['m']}")
    ax_left.set_xlabel('Time (t)')
    ax_left.set_ylabel('C(t)', color='blue')
    ax_right.set_ylabel('A(t)', color='red')

    ax_left.tick_params(axis='y', colors='blue')
    ax_right.tick_params(axis='y', colors='red')

    # Adding legends specifically to show the delay effect
    ax_left.legend(loc='upper left')
    ax_right.legend(loc='upper right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  # Add spacing between subplots
plt.show()
