import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
import math

tau = 0.56
p = 1
m = 5
b = 11.11
a_values = [0.6, 0.810, 1.215]
q = 0.03
mu = 0.03
Delta = 7.5

def adv1(Y, l, params):
    a = params['a']
    x = Y(l)[0]
    y = Y(l)[1]
    tau = q * Delta / a
    x_retraso = Y(l - tau)[1] if l - tau > 0 else Y(0)[1]    
    dxdl = 1 / (1 + (math.exp(-tau) * x_retraso)**m) - x
    dydl = (b * x - y) / p
    return np.array([dxdl, dydl])

l_span = (0, 100)
l_eval = np.linspace(*l_span, 1000)
x0 = 1
y0 = 0
y0_func = lambda l: np.array([x0, y0])

plt.figure(figsize=(10, 6))
for a in a_values:
    params = {'a': a}
    sol = ddeint(adv1, y0_func, l_eval, fargs=(params,))

    plt.plot(l_eval, sol[:, 0], label=f"x(l), a={a}")
    #plt.plot(l_eval, sol[:, 1], label=f"y(l), a={a}", linestyle='dashed')

plt.title("Solutions")
plt.xlabel("l")
plt.ylim(0, 0.4)
plt.ylabel("Values of x(l)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()