##Code for amp/period backup

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm


def calculate_amplitude_period(time, values):
    maxs = []
    maxs_times = []
    
    mins = []
    mins_times = []

    # Identify local maxima
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxs.append(values[i])
            maxs_times.append(time[i])

    #take sum min too lol
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            mins.append(values[i])
            mins_times.append(time[i])

    if len(maxs) < 2:
        return 0, 0  # Not enough oscillations to compute period

    if len(mins) < 2:
        return 0, 0  # Not enough oscillations to compute period

    amplitude = (maxs[-1] - mins[-1] ) / 2
    if maxs_times[-1] > mins_times[-1]:
        period = maxs_times[-1] - maxs_times[-2]
        # if maxs[-2] - maxs[-1] > 0.005:
        #     return 0, 0
    else:
        period = mins_times[-1] - mins_times[-2]
        # if mins[-2] - mins[-1] > 0.005:
        #     return 0, 0

    if amplitude <= 0.00001:
        return amplitude, 0
        
    return amplitude, period

def amp_values(time, values):
    maxima_indices, _ = find_peaks(values)
    minima_indices, _ = find_peaks(-values)

    nspeaks = np.concatenate((maxima_indices, minima_indices))

    peaks = np.sort(nspeaks)  ##esto son los INDICES, NO LOS VALORES DE values
    vpeaks = values[peaks]   ##esto son los valores maximos y minimos en orden

    amplitude_func = np.zeros(len(vpeaks)-1)

    for i in range(0, len(vpeaks)-1):
        amplitude_func[i] = np.abs((vpeaks[i+1] - vpeaks[i])/2)
    
    amp_times = np.linspace(0, time[-1], len(amplitude_func))

    return amp_times, amplitude_func


