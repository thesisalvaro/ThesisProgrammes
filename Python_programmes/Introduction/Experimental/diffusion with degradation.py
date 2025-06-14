import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
import math

#tau = 0.56
p = 1
m = 5
b = 11.11
D_values = [0.015, 0.03, 0.045]
q = 0.03
mu = 0.03
Delta = 7.5

def adv1(Y, l, params):
    D = params['D']
    x = Y(l)[0]
    y = Y(l)[1]
    tau = (q / mu) * (math.sqrt(9/4 + mu * (x**2 /D)) - 3/2)
    x_retraso = Y(l - tau)[1] if l - tau > 0 else Y(0)[1]    
    dxdl = 1 / (1 + (math.exp(-2*math.sqrt(x**2 * mu / D)) * x_retraso)**m) - x
    dydl = (b * x - y) / p
    return np.array([dxdl, dydl])

l_span = (0, 100)
l_eval = np.linspace(*l_span, 1000)
x0 = 1
y0 = 0
y0_func = lambda l: np.array([x0, y0])

plt.figure(figsize=(10, 6))
for D in D_values:
    params = {'D': D}
    sol = ddeint(adv1, y0_func, l_eval, fargs=(params,))

    plt.plot(l_eval, sol[:, 0], label=f"x(l), D={D}")
    #plt.plot(l_eval, sol[:, 1], label=f"y(l), D={D}", linestyle='dashed')

plt.title("Solutions")
plt.xlabel("l")
plt.ylim(0, 1.2)
plt.ylabel("Values of x(l)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()