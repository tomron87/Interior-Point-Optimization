import numpy as np

class InteriorPoint:
    def __init__(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
        self.current_t = 1.0
        self.mu = 10.0
        self.central_path = []
        
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
            'path': ([np.array(p[0]) for p in self.central_path], [p[1] for p in self.central_path]),
            'success': success
        }
    
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

