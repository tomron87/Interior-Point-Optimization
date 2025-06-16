import unittest
import numpy as np
from src.constrained_min import InteriorPoint
from tests.examples import quadratic, quadratic_ineq1, quadratic_ineq2, quadratic_ineq3, linear, linear_ineq1, linear_ineq2, linear_ineq3, linear_ineq4

class TestConstrainedMin(unittest.TestCase):
    def setUp(self):
        self.functions = [quadratic, linear]
        self.ineq_constraints = [[quadratic_ineq1, quadratic_ineq2, quadratic_ineq3], [linear_ineq1, linear_ineq2, linear_ineq3, linear_ineq4]]
        self.eq_constraints_mat = [np.array([[1, 1, 1]]), None]
        self.eq_constraints_rhs = [np.array([1]), None]
        self.x0 = [np.array([0.1, 0.2, 0.7]), np.array([0.5, 0.75])]
        self.plot_ranges = [(-0.5, 1.5), (-0.5, 1.5)]  # x and y ranges for plotting

    def test_qp(self):
        print("\nTesting Quadratic Programming Problem:")
        tol = 1e-8
        max_iter = 100
        x0 = self.x0[0]
        func = self.functions[0]
        ineq_constraints = self.ineq_constraints[0]
        eq_constraints_mat = self.eq_constraints_mat[0]
        eq_constraints_rhs = self.eq_constraints_rhs[0]
        
        minimizer = InteriorPoint(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)
        result = minimizer.minimize(x0, tol=tol, max_iter=max_iter)
        
        # Plot results
        minimizer.plot_feasible_region_and_path(self.plot_ranges[0], self.plot_ranges[1])
        minimizer.plot_objective_values()
        
        # Print final values
        minimizer.print_final_values(result['x'])
        
        # Assertions
        self.assertTrue(result['success'], "Interior point method failed to converge")
        self.assertTrue(np.isfinite(result['f']), "Interior point method produced non-finite result")
        self.assertTrue(np.allclose(result['x'], np.array([0, 0, 1]), atol=tol), "Interior point method did not converge to the correct solution")

    def test_lp(self):
        print("\nTesting Linear Programming Problem:")
        tol = 1e-8
        max_iter = 100
        x0 = self.x0[1]
        func = self.functions[1]
        ineq_constraints = self.ineq_constraints[1]
        eq_constraints_mat = self.eq_constraints_mat[1]
        eq_constraints_rhs = self.eq_constraints_rhs[1]
        
        minimizer = InteriorPoint(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)
        result = minimizer.minimize(x0, tol=tol, max_iter=max_iter)
        
        # Plot results
        minimizer.plot_feasible_region_and_path(self.plot_ranges[0], self.plot_ranges[1])
        minimizer.plot_objective_values()
        
        # Print final values
        minimizer.print_final_values(result['x'])
        
        # Assertions
        self.assertTrue(result['success'], "Interior point method failed to converge")
        self.assertTrue(np.isfinite(result['f']), "Interior point method produced non-finite result")
        self.assertTrue(np.allclose(result['x'], np.array([0, 0, 1]), atol=tol), "Interior point method did not converge to the correct solution")

if __name__ == '__main__':
    unittest.main()