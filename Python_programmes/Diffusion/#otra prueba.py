#otra prueba

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')  # or 'QtAgg' if available
from tqdm import tqdm
from Backup.code_backup import calculate_amplitude_period, amp_values


#--------------------------------------------------------

k1 = 0.08
K = 0.9
m = 10.0
q1 = 0.03
q2 = 0.03
k2 = 0.08
D = 0.1
mu = 0.03
Delta = 4.5

##PARAMETERS TO VARY ----------------------------------------

a_values = np.linspace(0.4, 0.425, 6)
m_values = np.linspace(7, 15, 160)
q2_values = [0.015, 0.02, 0.025, 0.03, 0.13, 0.135, 0.14, 0.145]
k1_values = [0.085, 0.09, 0.095, 0.1]
k2_values = [0.08, 0.085, 0.09, 0.095, 0.1]
K_values = [0.8]
Delta_values = np.linspace(2, 9, 140)
Delta_values2 = np.linspace(2, 11.2, 140)
#Delta_values = [4, 4.5, 5]
mu_values = [0.0, 0.03]
D_values = [0.06]

#------------------------------------------------------------

# Grid
C = 100.0                   # Spatial domain limit
Nx = 1001                   # Number of spatial points
x = np.linspace(0, C, Nx)  # Spatial grid
dx = x[1] - x[0]            # Spatial step

t_spanPDE = (0, 15000)           # Time range for simulation
dtPDE = 0.01     

tauPDE = np.zeros_like(Delta_values)
taudeg = np.zeros_like(tauPDE)

for i in range(0, len(Delta_values)):
    tauPDE[i] = (Delta_values[i])**2 /(6*D)
    taudeg[i] = (1/(4*mu)) *(-3 + np.sqrt(9 + (4*mu*Delta_values2[i]**2 )/(D))) 


print('esto es tau PDE', tauPDE)

print('-----------------')

print('esto es taudeg', taudeg)