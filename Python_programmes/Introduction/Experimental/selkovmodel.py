##nullclines of selkov model

##Used to investigate the selkov model, which ultimately did not make it to the thesis

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product

def nullclinex(x, params):
   a = params['a']
   nx = x / (a + x**2)
   return nx

def nullcliney(x, params):
   a = params['a']
   b = params['b']
   ny = b / (a + x**2)
   return ny

b_values = [0.1, 0.5]

a_values = [0.1, 0.1]

x = np. linspace(0, 1.5, 10000)

colors = ['b', 'g']

plt.figure(figsize=(12, 6))
kk = [0,1]

dot = '\u0307'

for a, b, colorao in zip(a_values, b_values, colors):
   params = {'a': a, 'b': b}
   plt.plot(x, nullcliney(x, params), color=colorao, label=f'y'+ dot + " = 0, b = {b}")

plt.plot(x, nullclinex(x, params), color = 'r', label=f'x' + dot + '= 0')

plt.xlabel('x')
plt.ylabel('y')
plt.title("Nullclines of the Selkov Model")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()




