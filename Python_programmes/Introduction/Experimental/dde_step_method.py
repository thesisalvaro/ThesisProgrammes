import numpy as np
from scipy.integrate import solve_ivp

# Define the DDE

def adv1_rhs(l, Y, delay_func, params):
    """
    Right-hand side of the advanced DDE system.

    Parameters:
    l : float
        Current independent variable.
    Y : ndarray
        Current state vector [x, y].
    delay_func : callable
        Function to get delayed values Y(l - tau).
    params : dict
        Parameters for the DDE, including 'a', 'q', 'Delta', 'm', 'b', and 'p'.

    Returns:
    dYdl : ndarray
        Derivative of the state vector [dx/dl, dy/dl].
    """
    a = params['a']
    q = params['q']
    Delta = params['Delta']
    m = params['m']
    b = params['b']
    p = params['p']

    tau = q * Delta / a
    x = Y[0]
    y = Y[1]

    if l - tau > 0:
        x_retraso = delay_func(l - tau)[1]
    else:
        x_retraso = delay_func(0)[1]

    dxdl = 1 / (1 + x_retraso**m) - x
    dydl = (b * x - y) / p

    return np.array([dxdl, dydl])

# Step method to solve advanced DDE
def solve_dde(func, g, tt, fargs=()):
    """
    Solve a delay differential equation system using the method of steps.

    Parameters:
    func : callable
        Right-hand side of the DDE (function of l, Y, delay_func, *fargs).
    g : callable
        History function Y(l) for l <= tt[0].
    tt : array-like
        Time points where the solution is desired.
    fargs : tuple, optional
        Additional arguments passed to func.

    Returns:
    y_values : ndarray
        Array of state vectors [x(l), y(l)] corresponding to tt.
    """
    l0 = tt[0]

    # Initialize storage for results
    l_values = [l0]
    y_values = [g(l0)]

    # Define a delay function
    def delay_func(l):
        if l < l0:
            return g(l)
        elif l > l_values[-1]:
            return y_values[-1]  # Extrapolate constant
        else:
            idx = np.searchsorted(l_values, l) - 1
            return y_values[idx]

    # Solve iteratively over time points
    for l_end in tt[1:]:
        l_start = l_values[-1]

        # Solve for the current interval
        sol = solve_ivp(
            lambda t, y: func(t, y, delay_func, *fargs),
            [l_start, l_end],
            y_values[-1],
            dense_output=True
        )

        # Store results
        l_values.extend(sol.t[1:])
        y_values.extend(sol.y[:, 1:].T)

    # Interpolate results to match tt
    y_interp = np.array([np.interp(tt, l_values, np.array(y_values)[:, i]) for i in range(len(y_values[0]))]).T
    return y_interp

# Example usage function (to be imported and used elsewhere)
def dde_solve(func, g, tt, fargs=()):
    """
    Wrapper for solve_dde with identical interface to ddeint.

    Parameters:
    func : callable
        Right-hand side of the DDE.
    g : callable
        History function.
    tt : array-like
        Time points where the solution is desired.
    fargs : tuple, optional
        Additional arguments passed to func.

    Returns:
    ndarray
        Solution of the DDE system.
    """
    return solve_dde(func, g, tt, fargs=fargs)
