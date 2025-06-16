import numpy as np

def quadratic(x):
    if (len(x) != 3):
        raise ValueError("x must be a 3D array")
    Q = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    v = np.array([0, 0, 1])
    c = 1
    f_x = 0.5 * x @ Q @ x.T + v @ x.T + c
    return [f_x, Q @ x + v, Q]

def quadratic_ineq1(x):
    if (len(x) != 3):
        raise ValueError("x must be a 3D array")
    ineq_vec = np.array([-1, 0, 0])
    return [ineq_vec @ x, ineq_vec, None]

def quadratic_ineq2(x):
    if (len(x) != 3):
        raise ValueError("x must be a 3D array")
    ineq_vec = np.array([0, -1, 0])
    return [ineq_vec @ x, ineq_vec, None]

def quadratic_ineq3(x):
    if (len(x) != 3):
        raise ValueError("x must be a 3D array")
    ineq_vec = np.array([0, 0, -1])
    return [ineq_vec @ x, ineq_vec, None]
    
def linear(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    A = np.array([-1, -1])
    return [A @ x, A, None]

def linear_ineq1(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    ineq_vec = np.array([-1, -1])
    return [ineq_vec @ x + 1, ineq_vec, None]

def linear_ineq2(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    ineq_vec = np.array([0, 1])
    return [ineq_vec @ x - 1, ineq_vec, None]

def linear_ineq3(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    ineq_vec = np.array([1, 0])
    return [ineq_vec @ x - 2, ineq_vec, None]

def linear_ineq4(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    ineq_vec = np.array([0, -1])
    return [ineq_vec @ x, ineq_vec, None]