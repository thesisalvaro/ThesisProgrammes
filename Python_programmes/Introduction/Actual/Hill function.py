## Hill function
## Press play

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

def X_equation(X, K, m):
    Y = (X**m)/(np.power(K,m)+np.power(X,m))
    return Y

X = np.linspace(0, 10, 100)
K = 5
m = 2
n_values = [1, 0.5, 2, 0.25, 4]
K_values = np.linspace(1, 5, 5)
m_values = [5, 8, 10, 15, 20 ,25]
m = 5

for i in range(0, len(m_values)):
    m = m_values[i]
    plt.plot(X, X_equation(X, K, m), label=f"m = {m}")

plt.xlabel("X")
plt.ylabel("f(X)")
plt.title("Hill function")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()