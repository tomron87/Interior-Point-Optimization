import numpy as np
import matplotlib.pyplot as plt

class InteriorPoint:
    def __init__(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
        self.current_t = 1.0
        self.mu = 10.0
        self.central_path = []
        self.objective_values = []
        
    def barrier_function(self, x, hessian_flag: bool = True):
        f_val, f_grad, f_hess = self.func(x)
        val = self.current_t * f_val
        grad = self.current_t * f_grad.copy()
        if hessian_flag:
            if f_hess is not None:
                hess = self.current_t * f_hess.copy()
            else:
                hess = np.zeros((x.size, x.size))
        
        for g in self.ineq_constraints:
            g_val, g_grad, g_hess = g(x)
            if g_val >= 0:
                if hessian_flag:
                    return np.inf, None, None
                return np.inf, None
            val -= np.log(-g_val)
            grad += -1.0 / g_val * g_grad
            if hessian_flag:
                hess += np.outer(g_grad, g_grad) / (g_val**2)
                if g_hess is not None:
                    hess -= g_hess / g_val
        if hessian_flag:
            return val, grad, hess
        return val, grad
            
    def backtracking(self, x, p, initial_alpha=1.0, c=0.01, rho=0.5, max_steps=20):
        alpha = initial_alpha
        for _ in range(max_steps):
            x_candidate = x + alpha * p
            candidate_val = self.barrier_function(x_candidate, hessian_flag=False)[0]
            fx, gx = self.barrier_function(x, hessian_flag=False)

            if candidate_val > fx + c * alpha * np.dot(gx, p) or np.isnan(candidate_val):
                print(f"[Backtracking] Wolfe not satisfied at alpha={alpha}, reducing alpha")
                alpha *= rho
            else:
                print(f"[Backtracking] Wolfe satisfied at alpha={alpha}")
                break
        
        if alpha < 1e-12:
            print("[Backtracking] Alpha too small; line search failed.")
            return 0.0

        return alpha

    def kkt_solver(self):
        if self.eq_constraints_mat is None or self.eq_constraints_mat.shape[0] == 0:
            # If there are no equality constraints, use plain Newton step direction
            def pk_resolver(x):
                _, gradient, hessian = self.barrier_function(x)
                try:
                    return -np.linalg.solve(hessian, gradient)
                except np.linalg.LinAlgError:
                    return -gradient
                
            return pk_resolver

        def pk_resolver(x, fallback_to_gd: bool = True):
            obj_val, gradient, hessian = self.func(x)
            A = self.eq_constraints_mat
            n = len(x)
            m = A.shape[0]
            KKT_mat = np.block([
                [hessian, A.T],
                [A, np.zeros((m, m))]
            ])
            rhs = np.concatenate([-gradient, np.zeros(m)])
            try:
                sol = np.linalg.solve(KKT_mat, rhs)
                p = sol[:n]
                if np.dot(gradient, p) >= 0:
                    raise ValueError
                return p
            except (np.linalg.LinAlgError, ValueError):
                if not fallback_to_gd:
                    raise
                z = -gradient
                ATA_inv = np.linalg.inv(A @ A.T)
                p_proj = z - A.T @ (ATA_inv @ (A @ z))
                return p_proj
        
        return pk_resolver
    

    def newton_step(self, 
                    x0, 
                    b_k, 
                    pk_resolver, 
                    alpha_resolver,
                    obj_tol=1e-8,
                    param_tol=1e-8,
                    max_iter=10000):
        x = x0
        success = False

        obj_val_k, gradient_k, hessian_k = self.barrier_function(x)

        for k in range(max_iter):
            rhs = pk_resolver(x)
            p_k = np.linalg.solve(b_k, rhs)
            alpha_k = self.backtracking(x, p_k)
            if alpha_k * np.linalg.norm(p_k) < param_tol:
                success = True
                break
            x_new = x + alpha_k * p_k
            obj_val_new, gradient_new, hessian_new = self.barrier_function(x_new)
            if abs(obj_val_new - obj_val_k) < obj_tol:
                x, obj_val_k = x_new, obj_val_new
                success = True
                break
            x, obj_val_k = x_new, obj_val_new
            
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
        success = False

        # Initialize tracking lists
        self.central_path = [(x.copy(), self.func(x)[0])]
        self.objective_values = [self.func(x)[0]]
        
        m = len(self.ineq_constraints) # number of inequality constraints

        if m and any(c(x)[0] >= 0 for c in self.ineq_constraints):
            raise ValueError("[InteriorPoint Minimizer]x0 is not strictly feasible (some g_i(x0) ≥ 0).")
        if self.eq_constraints_mat is not None and self.eq_constraints_mat.shape[0] > 0 and np.linalg.norm(self.eq_constraints_mat @ x - self.eq_constraints_rhs) > 1e-10:
            raise ValueError("[InteriorPoint Minimizer] x0 does not satisfy the equality constraints.")

        i = 0

        # Outer loop
        max_outer_iter = 100
        while i < max_outer_iter:
            phi_t = self.barrier_function(x)
            pk_resolver = self.kkt_solver()


            newton_result = self.newton_step(x, 
                                             np.identity(x.size), 
                                             pk_resolver, 
                                             self.backtracking)

            if not newton_result['success']:
                raise RuntimeError(f"Inner Newton failed to converge at iteration {i}")
            
            x = newton_result['x']
            self.central_path.append((x.copy(), self.func(x)[0]))
            self.objective_values.append(self.func(x)[0])
            
            if m == 0 or m / self.current_t < tol:
                print("[InteriorPoint Minimizer] Duality gap small enough – terminating.")
                success = True
                break
            
            self.current_t *= self.mu
            i += 1
        
        return {
            'x': x,
            'f': self.func(x)[0],
            'iter': i,
            'success': success
        }
         
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

