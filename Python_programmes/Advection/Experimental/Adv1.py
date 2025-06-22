## This program uses ddeint, which we ended up discarding cause it was not precise enough

##Was not really used, but helped us have an idea of what we were doing

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from itertools import product
from tqdm import tqdm


m_values = [5]
a_values = [0.5, 1, 1.5, 5, 10]
q1_values = [0.03]
q2_values = [0.03]
Delta_values = [7.5]
k3_values = [100]
q_values = [0.03]

l_span = (0, 200)
l_eval = np.linspace(*l_span, 50001)
t_eval = np.linspace(*l_span, 100)
x0 = 1
y0 = 0
z0 = 0
y0_func = lambda l: np.array([x0, y0])
y0_func2 = lambda l: np.array([x0, y0, z0])



def adv2(Y, l, params):
    a = params['a']
    q1 = params['q1']
    q2 = params['q2']
    Delta = params['Delta']
    m = params['m']
    k3 = params['k3']
    b = 1 / (k3 * q1 * q2)
    tau = q1 * Delta / a
    p = q1 / q2
    x = Y(l)[0]
    y = Y(l)[1]
    x_retraso = Y(l - tau)[1] if l - tau > 0 else Y(0)[1]
    dxdl = 1 / (1 + x_retraso**m) - x
    dydl = (b * x - y) / p
    return np.array([dxdl, dydl])


plt.figure(figsize=(10, 6))

for a, m, q1, q2, Delta, k3 in tqdm(product(a_values, m_values, q1_values, q2_values, Delta_values, k3_values)):
    params = {'a': a, 'm': m, 'q1': q1, 'q2':q2, 'Delta': Delta, 'k3': k3}
    sol = ddeint(adv2, y0_func, l_eval, fargs=(params,))
    tau = q1 * Delta / a
    b = 1 / (k3 * q1 * q2)
    p = q1 / q2
    plt.plot(l_eval, sol[:, 0], label=f"x(l), a={a}, m = {m}, p={p}, b={round(b, 2)}, q1={q1}, q2={q2}, Delta={Delta}, tau={round(tau, 2)}, K*={k3}") #This here plots x(l) ye
    #plt.plot(l_eval, sol[:, 1], label=f"y(l), a={a}, m = {m}", linestyle='dashed') #This here actually adds the y(x,t), so no interested rn

plt.title("Solutions")
plt.xlabel("l")
plt.ylim(0, 1.2)
plt.ylabel("Values of x(l)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()