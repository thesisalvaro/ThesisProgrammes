##Two-dimensional Goodwin Model

##Press play

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('QtAgg')  # or 'QtAgg' if available

def X_equation(x, y, k1, K, m, q1):
    return k1 * K/(np.power(K,m)+np.power(y,m))-q1

def Y_equation(x, y, k2, q2):
    return k2*x - q2

k1 = 0.1
K = 1.0
m = 1.0
q1 = 0.03
q2 = 0.03
k2 = 0.1
Delta = 7.5

m_values = [4.8, 4.9]
q2_values = [0.015, 0.02, 0.025, 0.03, 0.13, 0.135, 0.14, 0.145]
k1_values = [0.085, 0.09, 0.095, 0.1]
k2_values = [0.08, 0.085, 0.09, 0.095, 0.1]
K_values = [1, 1.025, 1.05, 1.075, 1.1, 1.125]
Delta_values = [100, 500, 1000]

t_span = (0, 5000)
dt = 0.02
t_steps = int(t_span[-1]/dt) 

X = np.zeros(t_steps)
Y = np.zeros([t_steps])
initial_valuesx = [0, 1, 2, 3]
initial_valuesy = [2, 0, 0, 1]


for j in range(0, len(initial_valuesx)):
    X[0] = initial_valuesx[j]
    Y[0] = initial_valuesy[j]

    for i in tqdm(range(1,t_steps)):
        k1_x = X_equation(X[i-1], Y[i-1], k1, K, m, q1)
        k1_y = Y_equation(X[i-1], Y[i-1], k2, q2)

        aux_x = X[i-1]+dt*k1_x
        aux_y = Y[i-1]+dt*k1_y
            
        k2_x = X_equation(aux_x, aux_y, k1, K, m, q1)
        k2_y = Y_equation(aux_x, aux_y, k2, q2)
            
        X[i] = X[i-1]+0.5*dt*(k1_x+k2_x)
        Y[i] = Y[i-1]+0.5*dt*(k1_y+k2_y)
    plt.plot(X, Y, label=f"(x0, y0) = ({X[0]}, {Y[0]})")


plt.xlabel("X(t)")
plt.ylabel("Y(t)")
plt.title("Phase portrait of the Goodwin Model")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

