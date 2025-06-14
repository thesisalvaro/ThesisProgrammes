## En este programa vamos a poner la ecuación de advección con su correspondiente solución y su ploteada 
## Este va a ser el pogramita weno weno y limpito
import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from itertools import product

#Parameters
m_values = [5]  #de 3 a 7
a_values = [0.405]  #de 0.2 a 0.6
q1_values = [0.03]
q2_values = [0.03]
Delta_values = [7.5]
k3_values = [100]
q_values = [0.03]
mu_values = [0.03]

l_span = (0, 150)
l_eval = np.linspace(*l_span, 20001)
t_eval = np.linspace(*l_span, 100)
x0 = 1
y0 = 0
z0 = 0
y0_func = lambda l: np.array([x0, y0])
y0_func2 = lambda l: np.array([x0, y0, z0])

##These are reference values used by Quentin in advection-only case
#tau = 0.56
#p = 1
#m = 5
#b = 11.11
#a_values = [0.405, 0.810, 1.215]
#q = 0.03
#Delta = 7.5

# def adv1(Y, l, params):
#     a = params['a']
#     q = params['q']
#     x = Y(l)[0]
#     y = Y(l)[1]
#     tau = q * Delta / a
#     b = 1 / (k3 * q1 * q2)
#     p = 1
#     x_retraso = Y(l - tau)[1] if l - tau > 0 else Y(0)[1]    
#     dxdl = 1 / (1 + x_retraso**m) - x
#     dydl = (b * x - y) / p
#     return np.array([dxdl, dydl])

## we have to transform these values to the adimensionalizated system
## Now this is full parametrized DDE

def adv2(Y, l, params):
    a = params['a']
    q1 = params['q1']
    q2 = params['q2']
    Delta = params['Delta']
    m = params['m']
    k3 = params['k3']
    mu = params['mu']
    b = 1 / (k3 * q1 * q2)
    tau = q1 * Delta / a
    tauprime = mu * Delta / a
    p = q1 / q2
    x = Y(l)[0]
    y = Y(l)[1]
    x_retraso = Y(l - tau)[1] if l - tau > 0 else Y(0)[1]
    dxdl = 1 / (1 + (np.exp(-tauprime) * x_retraso)**m) - x
    dydl = (b * x - y) / p
    return np.array([dxdl, dydl])


#for a in a_values:
#    params = {'a': a}
 #   sol = ddeint(adv1, y0_func, l_eval, fargs=(params,))
    #sol = stm(adv1, y0_func, l_eval, fargs=(params,))

  #  plt.plot(l_eval, sol[:, 0], label=f"x(l), a={a}")
   # plt.plot(l_eval, sol[:, 1], label=f"y(l), a={a}", linestyle='dashed')

plt.figure(figsize=(10, 6))

for a, m, q1, q2, Delta, k3, mu in product(a_values, m_values, q1_values, q2_values, Delta_values, k3_values, mu_values):
    params = {'a': a, 'm': m, 'q1': q1, 'q2':q2, 'Delta': Delta, 'k3': k3, 'mu': mu}
    sol = ddeint(adv2, y0_func, l_eval, fargs=(params,))
    tau = q1 * Delta / a
    b = 1 / (k3 * q1 * q2)
    p = q1 / q2
    tauprime = mu * Delta / a
    plt.plot(l_eval, sol[:, 0], label=f"x(l), a={a}, m = {m}, p={p}, b={round(b, 2)}, q1={q1}, q2={q2}, Delta={Delta}, tau={round(tau, 2)}, K*={k3}") #This here plots x(l) ye
    plt.plot(l_eval, sol[:, 1], label=f"y(l), a={a}, m = {m}", linestyle='dashed') #This here actually adds the y(x,t), so no interested rn

plt.title("Solutions")
plt.xlabel("l")
plt.ylim(0, 0.6)
plt.ylabel("Values of x(l)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()