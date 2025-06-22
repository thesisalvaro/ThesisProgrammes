import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
 
K_s = 1.4
b_deg = 0.1
K = 30

def hill_function(C, m):
    return C**m / (K**m + C**m)

def cellcycle(Y, t, params):
    C, A = Y(t)
    C_retraso = Y(t - params['tau'])[0] if t - params['tau'] > 0 else Y(0)[0]

    m = params['m']
    dCdt = K_s - b_deg * C * hill_function(C_retraso, m)
    dAdt = hill_function(C, m) - A
    return np.array([dCdt, dAdt])

initial_cond = [
    {'tau': 10, 'm': 10}, 
    {'tau': 10, 'm': 5},  
    {'tau': 5, 'm': 2}    
]
t_span = (0, 200)
t_eval = np.linspace(*t_span, 1000)
C0 = 1.0  ##Se asume que empezamos con concentración 1 entiendo
A0 = hill_function(C0, initial_cond[0]['m'])  
y0 = lambda t: np.array([C0, A0])
##print(A0)
"Hay que mirar si esto está bien o no porque no funca muy bien"

#fig, axs = plt.subplots(3, 1, figsize=(10, 12))
#for i, case in enumerate(cases):
 #   sol = ddeint(model, y0, t_eval, fargs=(case,))
#
 #   axs[i].plot(t_eval, sol[:, 0], label='C(t)', color='blue')  # C(t)
  #  axs[i].plot(t_eval, sol[:, 1], label='A(t)', color='red')   # A(t)

   # axs[i].set_title(f"Tau = {case['tau']}, m = {case['m']}")
   # axs[i].set_xlabel('Time (t)')
   # axs[i].set_ylabel('Concentration')
   # axs[i].legend()

#plt.tight_layout()
#plt.show()

"AAAAAAAAAAAAAAAAAAAAAAAa no puedo mas por dios"
fig, axs = plt.subplots(1, 3, figsize=(22, 4))
for i, case in enumerate(initial_cond):
    sol = ddeint(cellcycle, y0, t_eval, fargs=(case,))
    ax_left = axs[i]
    ax_right = ax_left.twinx() 
    ax_right.set_ylim(0, 1)
    ax_left.plot(t_eval, sol[:, 0], label='C(t)', color='blue')  
    ax_right.plot(t_eval, sol[:, 1], label='A(t)', color='red')  
    ax_left.set_title(f"Tau = {case['tau']}, m = {case['m']}")
    ax_left.set_xlabel('Time (t)')
    ax_left.set_ylabel('C(t)', color='blue')
    ax_right.set_ylabel('A(t)', color='red')
    ax_left.tick_params(axis='y', colors='blue')
    ax_right.tick_params(axis='y', colors='red')
    ax_left.legend(loc='upper left')
    ax_right.legend(loc='upper right')

plt.tight_layout()
plt.show()