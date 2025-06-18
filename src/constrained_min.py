import numpy as np
import matplotlib.pyplot as plt

class InteriorPoint:
    def __init__(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
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

            if g_val > 0:
                # If we're not feasible, skip this constraint
                continue
                            
            # Add to barrier value
            barrier_val -= np.log(-g_val)
            
            # Add to barrier gradient
            if g_grad is not None:
                barrier_grad += g_grad / (-g_val)
            
            # Add to barrier hessian
            if g_hess is not None:
                barrier_hess += (g_hess / (-g_val) + 
                            np.outer(g_grad, g_grad) / (g_val * g_val))
            else:
                barrier_hess += np.outer(g_grad, g_grad) / (g_val * g_val)

        
        # Combine objective and barrier terms
        total_val = t * f_val + barrier_val
        total_grad = t * f_grad + barrier_grad
        total_hess = t * f_hess + barrier_hess if f_hess is not None else barrier_hess
        
        return total_val, total_grad, total_hess
    
    def newton_step(self, x0, tol = 10e-12, max_iter = 1000):
        # Backtracking line search
        def backtracking(x, p, gradient, initial_alpha=1.0, c=0.01, rho=0.5):
            alpha = initial_alpha
            
            # First Wolfe condition (sufficient decrease)
            while self.func(x + alpha * p)[0] > self.func(x)[0] + c * alpha * np.dot(gradient, p):
                alpha = rho * alpha
                if alpha < 1e-10:  # Prevent too small step sizes
                    break
                    
            return alpha

        x = x0
        success = False

        obj_val_k, gradient_k, hessian_k = self.func(x)

        if hessian_k is None:
            raise ValueError("Hessian matrix is not provided")
                
        for k in range(max_iter):
            try:
                if self.eq_constraints_mat is not None and self.eq_constraints_mat.shape[0] > 0:
                    # Hard equality constraints: solve KKT
                    A = self.eq_constraints_mat
                    n = len(x)
                    m = A.shape[0]
                    KKT_mat = np.block([
                        [hessian_k, A.T],
                        [A, np.zeros((m, m))]
                    ])
                    rhs = np.concatenate([-gradient_k, np.zeros(m)])
                    sol = np.linalg.solve(KKT_mat, rhs)
                    p_k = sol[:n]
                else:
                    # Standard unconstrained Newton, with fallback to GD if Hessian is singular
                    try:
                        p_k = -np.linalg.solve(hessian_k, gradient_k)
                    except np.linalg.LinAlgError:
                        # If Hessian is singular, fall back to negative gradient direction
                        print(f"[Newton {k}] Hessian is singular, falling back to gradient descent")
                        p_k = -gradient_k

                alpha_k = backtracking(x, p_k, gradient_k)

                next_x = x + alpha_k * p_k
                next_obj_val = self.func(next_x)[0]

                if np.linalg.norm(next_x - x) < tol:
                    success = True
                    break
                if np.linalg.norm(next_obj_val - obj_val_k) < tol:
                    success = True
                    break

                x = next_x

            except Exception as e:
                print(f"[Newton {k}] Error: {e}")
                # Check if we're actually at optimum
                if np.linalg.norm(gradient_k) < tol:
                    return {
                        'x': x,
                        'f': obj_val_k,
                        'iter': k,
                        'success': True
                    }
                return {
                    'x': x,
                    'f': obj_val_k,
                    'iter': k,
                    'success': False
                }

        return {
            'x': x,
            'f': obj_val_k,
            'iter': k,
            'success': success
        }

    def minimize(self, x0, tol=1e-12):
        """
        Solve the constrained optimization problem using interior point method
        """
        x = x0.copy()
        t = 1.0
        mu = 10.0

        # Initialize tracking lists
        self.central_path = [(x.copy(), self.func(x)[0])]
        self.objective_values = [self.func(x)[0]]
        
        i = 0

        # Outer loop
        max_outer_iter = 100
        while i < max_outer_iter:
            # Create barrier function for current t
            def barrier_obj(x):
                return self.barrier_function(x, t)
            
            # Inner loop
            newton_result = self.newton_step(x, tol)
                        
            if not newton_result['success']:
                print(f"\nInner minimization failed at iteration {i}")
                return {
                    'x': x,
                    'f': self.func(x)[0],
                    'iter': i,
                    'success': False
                }
                
            x = newton_result['x']
            
            # Track central path and objective values
            self.central_path.append((x.copy(), self.func(x)[0]))
            self.objective_values.append(self.func(x)[0])
            
            # Check equality constraint violation
            eq_violation = 0
            if self.eq_constraints_mat is not None and len(self.eq_constraints_mat) > 0:
                eq_residual = self.eq_constraints_mat @ x - self.eq_constraints_rhs
                eq_violation = np.linalg.norm(eq_residual)
            
            # Check if we're close enough to the solution
            if len(self.ineq_constraints) / t < tol and eq_violation < tol:
                print("\nConverged! All constraints satisfied within tolerance.")
                return {
                    'x': x,
                    'f': self.func(x)[0],
                    'iter': i,
                    'success': True
                }
                
            t *= mu
            i += 1  
    
        print("\nReached maximum iterations without convergence")
        # If the loop exits due to max_outer_iter, check feasibility
        eq_violation = 0
        if self.eq_constraints_mat is not None and len(self.eq_constraints_mat) > 0:
            eq_residual = self.eq_constraints_mat @ x - self.eq_constraints_rhs
            eq_violation = np.linalg.norm(eq_residual)
        if all(g(x)[0] < -1e-8 for g in self.ineq_constraints) and eq_violation < tol:
            return {'x': x, 'f': self.func(x)[0], 'iter': i, 'success': True}
        else:
            return {'x': x, 'f': self.func(x)[0], 'iter': i, 'success': False}

    def plot_feasible_region_and_path(self, x_range, y_range, num_points=100):
        """Plot the feasible region and central path"""
        plt.figure(figsize=(10, 8))
        
        # Create grid for plotting
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        
        # Determine problem dimension from first constraint
        try:
            # Try with 3D point first
            test_point = np.zeros(3)
            g_val, g_grad, _ = self.ineq_constraints[0](test_point)
            is_3d = True
        except ValueError:
            # If that fails, it's a 2D problem
            is_3d = False
        
        # Plot inequality constraints
        for g in self.ineq_constraints:
            Z = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    point = np.array([X[i,j], Y[i,j]])
                    if is_3d:
                        point = np.pad(point, (0, 1))  # Add zero for z-coordinate
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
        
        # Plot central path (projected to 2D if needed)
        path_points = np.array([point[:2] for point, _ in self.central_path])
        plt.plot(path_points[:,0], path_points[:,1], 'g-', label='Central Path', linewidth=2)
        plt.plot(path_points[-1,0], path_points[-1,1], 'ro', label='Final Solution')
        
        plt.title('Feasible Region and Central Path')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.pause(0.1)  # Small pause to ensure plot is displayed

    def plot_objective_values(self):
        """Plot objective values vs iteration number"""
        plt.figure(figsize=(10, 6))
        iterations = range(len(self.objective_values))
        plt.plot(iterations, self.objective_values, 'b-', linewidth=2)
        plt.title('Objective Value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.grid(True)
        plt.draw()
        plt.pause(0.1)  # Small pause to ensure plot is displayed

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

