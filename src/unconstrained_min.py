import numpy as np

class LineSearch:
    def __init__(self, obj_func):
        """
        Initialize the unconstrained minimizer.
        
        Parameters:
        -----------
        obj_func : callable
            The objective function to minimize
        grad_func : callable, optional
            The gradient function of the objective function
        hessian_func : callable, optional
            The Hessian function of the objective function
        """
        self.obj_func = obj_func
        self.grad_val = []
        self.hessian_val = []
        self.path = []
    
    def minimize(self, x0, method='GD', obj_tol=1e-6, param_tol=1e-6, max_iter=1000):
        """
        Minimize the objective function starting from point x0.
        
        Parameters:
        -----------
        x0 : numpy.ndarray
            Initial point
        method : str
            Optimization method to use - GD for gradient descent, Newton for Newton's method
        obj_tol : float
            Objective function tolerance for convergence
        param_tol : float
            Parameter tolerance for convergence
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        dict
            Results of the optimization including:
            - x: the minimizer (final location)
            - f: the minimum value (final objective function value)
            - success: whether optimization was successful
        """
        if method == 'GD':
            return self.gradient_descent(x0, obj_tol, param_tol, max_iter)
        elif method == 'Newton':
            return self.newton(x0, obj_tol, param_tol, max_iter)
        else:
            raise ValueError(f"Invalid method: {method}")

    def gradient_descent(self, x0, obj_tol, param_tol, max_iter):
        """
        Gradient descent algorithm.

        Parameters:
        -----------
        x0 : numpy.ndarray
            Initial point
        obj_tol : float
            Objective function tolerance for convergence
        param_tol : float
            Parameter tolerance for convergence
        max_iter : int
            Maximum number of iterations

        Returns:
        --------
        dict
            Results of the optimization including:
            - x: the minimizer (final location)
            - f: the minimum value (final objective function value)
            - success: whether optimization was successful
        """
        x = x0
        success = False
        B_k = np.eye(len(x0))
        self.path = [(x.copy(), self.obj_func(x)[0])]
        for k in range(max_iter):
            gradient_k = self.obj_func(x)[1]
            p_k = -B_k @ gradient_k
            alpha_k = self.backtracking(x, p_k, gradient_k)
            obj_val_k = self.obj_func(x)[0]
            self.grad_val.append(gradient_k)
            next_x = x + alpha_k * p_k
            self.path.append((next_x.copy(), self.obj_func(next_x)[0]))
            if np.linalg.norm(next_x - x) < param_tol:
                success = True
                break
            if np.linalg.norm(self.obj_func(next_x)[0] - obj_val_k) < obj_tol:
                success = True
                break
            x = next_x
        
        return {
            'x': x,
            'f': self.obj_func(x)[0],
            'iter': k,
            'success': success
        }
      
    def newton(self, x0, obj_tol, param_tol, max_iter):
        """
        Newton's method.

        Parameters:
        -----------
        x0 : numpy.ndarray
            Initial point
        obj_tol : float
            Objective function tolerance for convergence
        param_tol : float
            Parameter tolerance for convergence
        max_iter : int
            Maximum number of iterations

        Returns:
        --------
        dict
            Results of the optimization including:
            - x: the minimizer (final location)
            - f: the minimum value (final objective function value)
            - success: whether optimization was successful
        """
        if (self.obj_func(x0)[2] is None):
            raise ValueError("Hessian matrix is not provided")
        
        x = x0
        success = False
        self.path = [(x.copy(), self.obj_func(x)[0])]
        for k in range(max_iter):
            hessian_k = self.obj_func(x)[2]
            gradient_k = self.obj_func(x)[1]
            p_k = -np.linalg.inv(hessian_k) @ gradient_k
            alpha_k = self.backtracking(x, p_k, gradient_k)
            obj_val_k = self.obj_func(x)[0]
            self.grad_val.append(gradient_k)
            self.hessian_val.append(hessian_k)
            next_x = x + alpha_k * p_k
            self.path.append((next_x.copy(), self.obj_func(next_x)[0]))
            if np.linalg.norm(next_x - x) < param_tol:
                success = True
                break
            if np.linalg.norm(self.obj_func(next_x)[0] - obj_val_k) < obj_tol:
                success = True
                break
            x = next_x
        
        return {
            'x': x,
            'f': self.obj_func(x)[0],
            'iter': k,
            'success': success
        }

    def backtracking(self, x, p, gradient, initial_alpha=1.0, c=0.01, rho=0.5):
        """
        Backtracking line search algorithm with Wolfe conditions.

        Parameters:
        -----------
        x : numpy.ndarray
            Current point
        p : numpy.ndarray
            Search direction
        gradient : numpy.ndarray
            Gradient at current point
        initial_alpha : float, optional
            Initial step size
        c : float, optional
            Sufficient decrease parameter
        rho : float, optional
            Step size reduction factor

        Returns:
        --------
        alpha : float
            Step size
        """
        alpha = initial_alpha
        
        # First Wolfe condition (sufficient decrease)
        while self.obj_func(x + alpha * p)[0] > self.obj_func(x)[0] + c * alpha * np.dot(gradient, p):
            alpha = rho * alpha
            if alpha < 1e-10:  # Prevent too small step sizes
                break
                
        return alpha



