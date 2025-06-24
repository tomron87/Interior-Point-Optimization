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
            
    def backtracking(self, x, p, initial_alpha=1.0, c=0.5, rho=0.5, max_steps=30):
        alpha = initial_alpha
        fx, gx = self.barrier_function(x, hessian_flag=False)
        last_feasible_alpha = None
        last_feasible_val = None
        for _ in range(max_steps):
            x_candidate = x + alpha * p
            candidate_val, _ = self.barrier_function(x_candidate, hessian_flag=False)
            if np.isnan(candidate_val) or np.isinf(candidate_val):
                alpha *= rho
                continue
            if candidate_val <= fx + c * alpha * np.dot(gx, p):
                return alpha
            # If function value decreased and candidate is feasible, remember it
            if candidate_val < fx:
                last_feasible_alpha = alpha
                last_feasible_val = candidate_val
            alpha *= rho
        # After max_steps, take last feasible step if it at least decreases the function
        if last_feasible_alpha is not None:
            print("[Backtracking] Accepting last feasible alpha that decreases fx:", last_feasible_alpha)
            return last_feasible_alpha
        print("[Backtracking] Line search failed; no feasible decrease found.")
        return 0.0

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
            print(f"[Newton {k}] dot(gx, p_k) = {np.dot(gradient_k, p_k)}")
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
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.patches import Polygon

        def find_feasible_polygon(x_bounds, y_bounds, num_points=100):
            """Compute vertices of the feasible region (intersection of half-planes)."""
            # Generate a grid of points over the plotting area
            x = np.linspace(x_bounds[0], x_bounds[1], num_points)
            y = np.linspace(y_bounds[0], y_bounds[1], num_points)
            X, Y = np.meshgrid(x, y)
            points = np.vstack([X.ravel(), Y.ravel()]).T

            # Keep only points satisfying ALL constraints
            mask = np.ones(points.shape[0], dtype=bool)
            for g in self.ineq_constraints:
                mask &= np.array([g(pt)[0] <= 0 for pt in points])
            feasible_points = points[mask]

            if len(feasible_points) < 3:
                return None  # Not enough points to form a polygon

            # Order points to form the polygon (ConvexHull)
            from scipy.spatial import ConvexHull
            hull = ConvexHull(feasible_points)
            polygon = feasible_points[hull.vertices]
            return polygon
        
        try:
            test_point = np.zeros(3)
            g_val, g_grad, _ = self.ineq_constraints[0](test_point)
            is_3d = True
        except Exception:
            is_3d = False

        if is_3d:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            # Plot the feasible triangle for the standard QP (simplex)
            verts = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            tri = Poly3DCollection([verts], alpha=0.2, facecolor='blue')
            ax.add_collection3d(tri)
            # Plot the central path
            path_points = np.array([point for point, _ in self.central_path])
            ax.plot(path_points[:,0], path_points[:,1], path_points[:,2], 'o-', color='tab:blue', label='central path')
            ax.scatter(path_points[-1,0], path_points[-1,1], path_points[-1,2], color='red', marker='*', s=200, label='final sol.')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('QP - feasible triangle & central path')
            ax.legend()
            ax.grid(True)
            return fig

        else:
            # 2D case
            fig, ax = plt.subplots(figsize=(10, 8))
            x = np.linspace(x_range[0], x_range[1], num_points)
            y = np.linspace(y_range[0], y_range[1], num_points)
            X, Y = np.meshgrid(x, y)
            polygon = find_feasible_polygon(x_range, y_range)
            if polygon is not None:
                poly_patch = Polygon(polygon, color='gray', alpha=0.3, zorder=1)
                ax.add_patch(poly_patch)
            for g in self.ineq_constraints:
                Z = np.zeros_like(X)
                for i in range(len(x)):
                    for j in range(len(y)):
                        point = np.array([X[i,j], Y[i,j]])
                        Z[i,j] = g(point)[0]
                ax.contour(X, Y, Z, levels=[0], colors='blue', alpha=0.5)
                # Optionally fill the feasible region using the intersection polygon for clarity

            # Plot equality constraints
            if self.eq_constraints_mat is not None:
                for i in range(len(self.eq_constraints_mat)):
                    a, b = self.eq_constraints_mat[i][:2]
                    c = self.eq_constraints_rhs[i]
                    if b != 0:
                        y_eq = (c - a*x) / b
                        ax.plot(x, y_eq, 'r--', label=f'Eq {i+1}')
            # Plot central path (projected to 2D)
            path_points = np.array([point[:2] for point, _ in self.central_path])
            ax.plot(path_points[:,0], path_points[:,1], 'g-', label='Central Path', linewidth=2)
            ax.plot(path_points[-1,0], path_points[-1,1], 'ro', label='Final Solution')
            ax.set_title('Feasible Region and Central Path')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True)
            return fig
    
    def plot_objective_values(self):
        """Plot objective values vs iteration number and return the matplotlib Figure object."""
        fig, ax = plt.subplots(figsize=(10, 6))
        iterations = range(len(self.objective_values))
        ax.plot(iterations, self.objective_values, 'b-', linewidth=2)
        ax.set_title('Objective Value vs Iteration')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.grid(True)
        # Do not show or pause, just return the figure
        return fig

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

