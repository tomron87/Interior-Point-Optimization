import numpy as np
import matplotlib.pyplot as plt
from unconstrained_min import LineSearch

class InteriorPoint:
    def __init__(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
        self.x0 = x0
        self.central_path = []  # Store points along central path
        self.objective_values = []  # Store objective values
        
    def barrier_function(self, x, t):
        # Get objective function values
        f_val, f_grad, f_hess = self.func(x)
        
        # Initialize barrier terms
        barrier_val = 0
        barrier_grad = np.zeros_like(x)
        barrier_hess = np.zeros((len(x), len(x)))
        
        # Add contribution from each inequality constraint
        for g in self.ineq_constraints:
            g_val, g_grad, g_hess = g(x)
            
            if g_val > 0:  # If constraint is violated
                return np.inf, None, None
                
            # Add to barrier value
            barrier_val -= np.log(-g_val)
            
            # Add to barrier gradient
            if g_grad is not None:
                barrier_grad += g_grad / (-g_val)
            
            # Add to barrier hessian
            if g_hess is not None: 
                barrier_hess += (g_hess / (-g_val) + 
                            np.outer(g_grad, g_grad) / (g_val * g_val))
        
        # Add equality constraints using quadratic penalty
        if self.eq_constraints_mat is not None and len(self.eq_constraints_mat) > 0:
            # Compute Ax - b
            eq_residual = self.eq_constraints_mat @ x - self.eq_constraints_rhs
            
            # Add penalty term to value
            barrier_val += t * np.sum(eq_residual ** 2)
            
            # Add penalty term to gradient
            barrier_grad += 2 * t * (self.eq_constraints_mat.T @ eq_residual)
            
            # Add penalty term to hessian
            barrier_hess += 2 * t * (self.eq_constraints_mat.T @ self.eq_constraints_mat)
        
        # Combine objective and barrier terms
        total_val = t * f_val + barrier_val
        total_grad = t * f_grad + barrier_grad
        total_hess = t * f_hess + barrier_hess
        
        return total_val, total_grad, total_hess
    
    def plot_feasible_region_and_path(self, x_range, y_range, num_points=100):
        """Plot the feasible region and central path"""
        plt.figure(figsize=(10, 8))
        
        # Create grid for plotting
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        
        # Plot inequality constraints
        for g in self.ineq_constraints:
            Z = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    point = np.array([X[i,j], Y[i,j]])
                    Z[i,j] = g(point)[0]
            plt.contour(X, Y, Z, levels=[0], colors='blue', alpha=0.5)
            plt.fill_between(x, y_range[0], y_range[1], where=Z[0,:] <= 0, color='blue', alpha=0.1)
        
        # Plot equality constraints
        if self.eq_constraints_mat is not None:
            for i in range(len(self.eq_constraints_mat)):
                a, b = self.eq_constraints_mat[i][:2]
                c = self.eq_constraints_rhs[i]
                if b != 0:
                    y_eq = (c - a*x) / b
                    plt.plot(x, y_eq, 'r--', label=f'Eq {i+1}')
        
        # Plot central path
        path_points = np.array([point for point, _ in self.central_path])
        plt.plot(path_points[:,0], path_points[:,1], 'g-', label='Central Path', linewidth=2)
        plt.plot(path_points[-1,0], path_points[-1,1], 'ro', label='Final Solution')
        
        plt.title('Feasible Region and Central Path')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_objective_values(self):
        """Plot objective values vs iteration number"""
        plt.figure(figsize=(10, 6))
        iterations = range(len(self.objective_values))
        plt.plot(iterations, self.objective_values, 'b-', linewidth=2)
        plt.title('Objective Value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.grid(True)
        plt.show()

    def print_final_values(self, x):
        """Print final objective and constraint values"""
        f_val, _, _ = self.func(x)
        print("\nFinal Results:")
        print(f"Objective value: {f_val:.6f}")
        
        print("\nInequality Constraints:")
        for i, g in enumerate(self.ineq_constraints):
            g_val, _, _ = g(x)
            print(f"g{i+1}(x) = {g_val:.6f}")
        
        if self.eq_constraints_mat is not None:
            print("\nEquality Constraints:")
            eq_residual = self.eq_constraints_mat @ x - self.eq_constraints_rhs
            for i, res in enumerate(eq_residual):
                print(f"h{i+1}(x) = {res:.6f}")

    def minimize(self, x0, tol=1e-8, max_iter=100):
        """
        Solve the constrained optimization problem using interior point method
        """
        x = x0.copy()
        t = 1.0
        mu = 10.0
        
        # Initialize tracking lists
        self.central_path = [(x.copy(), self.func(x)[0])]
        self.objective_values = [self.func(x)[0]]
        
        # Outer loop
        for i in range(max_iter):
            # Create barrier function for current t
            def barrier_obj(x):
                return self.barrier_function(x, t)
            
            # Create unconstrained minimizer
            minimizer = LineSearch(barrier_obj)
            
            # Inner loop
            result = minimizer.minimize(
                x,
                method='Newton',
                obj_tol=tol,
                param_tol=tol,
                max_iter=1000
            )
            
            if not result['success']:
                return {
                    'x': x,
                    'f': self.func(x)[0],
                    'iter': i,
                    'success': False
                }
                
            x = result['x']
            
            # Track central path and objective values
            self.central_path.append((x.copy(), self.func(x)[0]))
            self.objective_values.append(self.func(x)[0])
            
            # Check convergence
            eq_violation = 0
            if self.eq_constraints_mat is not None and len(self.eq_constraints_mat) > 0:
                eq_residual = self.eq_constraints_mat @ x - self.eq_constraints_rhs
                eq_violation = np.linalg.norm(eq_residual)
            
            if len(self.ineq_constraints) / t < tol and eq_violation < tol:
                return {
                    'x': x,
                    'f': self.func(x)[0],
                    'iter': i,
                    'success': True
                }
                
            t *= mu
        
        return {
            'x': x,
            'f': self.func(x)[0],
            'iter': i,
            'success': False
        }
