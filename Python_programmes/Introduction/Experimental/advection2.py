##Esto es una prueba para resolver Goodwin model de mierda usando múltiples parámetros (cuando digo múltiples
## me refiero a tela) así que vamo a darle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from ddeint import ddeint
from itertools import product

#tau = 0.56
p = 1
#m = 5
b = 11.11
#a_values = [0.405, 0.810, 1.215]
q = 0.03
Delta = 7.5

def goodwin(Y, l, params):
    K = params['K']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    q1 = params['q1']
    q2 = params['q2']
    q3 = params['q3']
    x = Y(l)[0]
    y = Y(l)[1]
    z = Y(l)[2]
    dxdl = k1 / (K**m + z**m) - q1 * x
    dydl = k2 * x - q2 * y
    dzdl = k3 * y - q3 * z
    return np.array([dxdl, dydl, dzdl])


def adv1(Y, l, params):
    a = params['a']
    x = Y(l)[0]
    y = Y(l)[1]
    tau = q * Delta / a
    x_retraso = Y(l - tau)[1] if l - tau > 0 else Y(0)[1]    
    dxdl = 1 / (1 + x_retraso**m) - x
    dydl = (b * x - y) / p
    return np.array([dxdl, dydl])


tau_values = [0.56]
p_values = [1]
m_values = [5]
b_values = [11.11]
a_values = [0.405]
q_values = [0.03]
Delta_values = [7.5]


## we have to transform these values to the adimensionalizated system

def adv2(Y, l, params):
    a = params['a']
    q = params['q']
    Delta = params['Delta']
    m = params['m']
    k3 = params['k3']
    b = k3 / (q ** 2)
    tau = q * Delta / a
    x = Y(l)[0]
    y = Y(l)[1]
    x_retraso = Y(l - tau)[1] if l - tau > 0 else Y(0)[1]
    dxdl = 1 / (1 + x_retraso**m) - x
    dydl = (b * x - y) / p
    return np.array([dxdl, dydl])

l_span = (0, 100)
l_eval = np.linspace(*l_span, 10000)
t_eval = np.linspace(*l_span, 100)
x0 = 1
y0 = 0
z0 = 0
y0_func = lambda l: np.array([x0, y0])
y0_func2 = lambda l: np.array([x0, y0, z0])


#print(np.array([x0, y0]))

plt.figure(figsize=(10, 6))
#for a in a_values:
#    params = {'a': a}
 #   sol = ddeint(adv1, y0_func, l_eval, fargs=(params,))
    #sol = stm(adv1, y0_func, l_eval, fargs=(params,))

  #  plt.plot(l_eval, sol[:, 0], label=f"x(l), a={a}")
   # plt.plot(l_eval, sol[:, 1], label=f"y(l), a={a}", linestyle='dashed')

heatmap_data = []

for a, m, p, b, q, Delta in product(a_values, m_values, p_values, b_values, q_values, Delta_values):
    params = {'a': a, 'm': m, 'p': p, 'b': b, 'q': q, 'Delta': Delta}
    sol = ddeint(adv2, y0_func, l_eval, fargs=(params,))
    tau = q * Delta / a
    plt.plot(l_eval, sol[:, 0], label=f"x(l), a={a}, m = {m}, p={p}, b={b}, q={q}, Delta={Delta}, tau={round(tau, 2)}")
    heatmap_data.append(sol[:, 0])
    #plt.plot(l_eval, sol[:, 1], label=f"y(l), a={a}, m = {m}", linestyle='dashed')

heatmap_data = np.array(heatmap_data)

#print(heatmap_data)

t = l_eval
X = sol[0, :]
Y = sol[1:, :]    

plt.title("Solutions")
plt.xlabel("l")
plt.ylim(0, 0.35)
plt.ylabel("Values of x(l) and y(l)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

exit()

##heatmap adsdjsiadjaidjsaidj
plt.figure(figsize=(10, 6))
plt.imshow(heatmap_data, extent=[l_span[0], l_span[1], a_values[0], a_values[-1]], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='x(l)')
plt.xlabel('Time-like variable (l)')
plt.ylabel('Parameter a')
plt.title('Heatmap of x(l) for varying a and l')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(heatmap_data, extent=[m_values[0], m_values[-1], a_values[0], a_values[-1]], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='x(l)')
plt.xlabel('Parameter m')
plt.ylabel('Parameter a')
plt.title('Amplitude Heatmap of x(l) in (a, m) Phase Space')
plt.show()

##Mathematical delayed systems can create oscillations --> how?? 

