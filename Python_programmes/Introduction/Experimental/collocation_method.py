import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from dde_step_method import solve_dde

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

# Collocation method to solve DDE
def collocation_dde_solver(func, g, tt, fargs=(), num_collocation_points=5):
    """
    Solve a delay differential equation system using the collocation method.

    Parameters:
    func : callable
        Right-hand side of the DDE (function of l, Y, delay_func, *fargs).
    g : callable
        History function Y(l) for l <= tt[0].
    tt : array-like
        Time points where the solution is desired.
    fargs : tuple, optional
        Additional arguments passed to func.
    num_collocation_points : int, optional
        Number of collocation points in each interval.

    Returns:
    y_values : ndarray
        Array of state vectors [x(l), y(l)] corresponding to tt.
    """
    l0 = tt[0]

    # Initialize storage for results
    y_values = [g(l0)]
    l_values = [l0]

    # Define a delay function using interpolation
    def delay_func(l):
        if l < l0:
            return g(l)
        elif l > l_values[-1]:
            return y_values[-1]  # Extrapolate constant
        else:
            interp = interp1d(l_values, np.array(y_values), axis=0, fill_value="extrapolate")
            return interp(l)

    # Solve iteratively over time points
    for i in range(1, len(tt)):
        l_start = tt[i - 1]
        l_end = tt[i]
        collocation_points = np.linspace(l_start, l_end, num_collocation_points)

        # Initialize with the last known value
        Y_collocation = np.zeros((num_collocation_points, len(y_values[0])))
        Y_collocation[0] = y_values[-1]

        # Solve at collocation points
        for j in range(1, num_collocation_points):
            l_coll = collocation_points[j]

            sol = solve_ivp(
                lambda t, y: func(t, y, delay_func, *fargs),
                [collocation_points[j - 1], l_coll],
                Y_collocation[j - 1],
                method='RK45',
                t_eval=[l_coll]
            )

            Y_collocation[j] = sol.y[:, -1]

        # Append results
        l_values.extend(collocation_points[1:])
        y_values.extend(Y_collocation[1:])

    # Interpolate results to match tt
    y_interp = np.array([np.interp(tt, l_values, np.array(y_values)[:, i]) for i in range(len(y_values[0]))]).T
    return y_interp

# Example usage function (to be imported and used elsewhere)
def dde_solve(func, g, tt, fargs=(), method="steps", **kwargs):
    """
    Solve a DDE system using the specified method.

    Parameters:
    func : callable
        Right-hand side of the DDE.
    g : callable
        History function.
    tt : array-like
        Time points where the solution is desired.
    fargs : tuple, optional
        Additional arguments passed to func.
    method : str, optional
        Method to use for solving the DDE ('steps' or 'collocation').
    kwargs : dict, optional
        Additional arguments for the selected method.

    Returns:
    ndarray
        Solution of the DDE system.
    """
    if method == "steps":
        return solve_dde(func, g, tt, fargs=fargs)
    elif method == "collocation":
        return collocation_dde_solver(func, g, tt, fargs=fargs, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
