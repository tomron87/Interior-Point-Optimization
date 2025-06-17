import unittest
import numpy as np
import sys
import os
from datetime import datetime
from src.constrained_min import InteriorPoint
from tests.examples import quadratic, quadratic_ineq1, quadratic_ineq2, quadratic_ineq3, linear, linear_ineq1, linear_ineq2, linear_ineq3, linear_ineq4

class TestConstrainedMin(unittest.TestCase):
    def setUp(self):
        # Create logs directory if it doesn't exist
        self.logs_dir = 'logs'
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        # Set up logging to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.logs_dir, f'test_results_{timestamp}.log')
        self.original_stdout = sys.stdout
        self.log = open(self.log_file, 'w')
        sys.stdout = self.log
        
        # Test setup
        self.functions = [quadratic, linear]
        self.ineq_constraints = [[quadratic_ineq1, quadratic_ineq2, quadratic_ineq3], [linear_ineq1, linear_ineq2, linear_ineq3, linear_ineq4]]
        self.eq_constraints_mat = [np.array([[1, 1, 1]]), None]
        self.eq_constraints_rhs = [np.array([1]), None]
        self.x0 = [np.array([0.1, 0.2, 0.7]), np.array([0.5, 0.75])]
        self.plot_ranges = [(-0.5, 1.5), (-0.5, 1.5)]  # x and y ranges for plotting

    def tearDown(self):
        # Restore stdout and close log file
        sys.stdout = self.original_stdout
        self.log.close()
        print(f"Test results have been logged to: {self.log_file}")

    def test_qp(self):
        print("\nTesting Quadratic Programming Problem:")
        tol = 1e-12
        max_iter = np.inf
        x0 = self.x0[0]
        func = self.functions[0]
        ineq_constraints = self.ineq_constraints[0]
        eq_constraints_mat = self.eq_constraints_mat[0]
        eq_constraints_rhs = self.eq_constraints_rhs[0]

        print("Initial inequality constraint values (QP):")
        for i, g in enumerate(ineq_constraints):
            print(f"g{i}(x0) = {g(x0)[0]}")
        
        minimizer = InteriorPoint(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)
        result = minimizer.minimize(x0, tol=tol, max_iter=max_iter)
        
        # Plot results
        minimizer.plot_feasible_region_and_path(self.plot_ranges[0], self.plot_ranges[1])
        minimizer.plot_objective_values()
        
        # Print final values
        minimizer.print_final_values(result['x'])

        print("Final result:", result)
        print("Final x:", result['x'])
        print("Final f(x):", result['f'])
        
        # Assertions
        self.assertTrue(result['success'], "Interior point method failed to converge")
        self.assertTrue(np.isfinite(result['f']), "Interior point method produced non-finite result")
        self.assertTrue(np.allclose(result['x'], np.array([0, 0, 1]), atol=tol), "Interior point method did not converge to the correct solution")

    def test_lp(self):
        print("\nTesting Linear Programming Problem:")
        tol = 1e-12
        max_iter = np.inf
        x0 = self.x0[1]
        func = self.functions[1]
        ineq_constraints = self.ineq_constraints[1]
        eq_constraints_mat = self.eq_constraints_mat[1]
        eq_constraints_rhs = self.eq_constraints_rhs[1]

        print("Initial inequality constraint values (LP):")
        for i, g in enumerate(ineq_constraints):
            print(f"g{i}(x0) = {g(x0)[0]}")
        
        minimizer = InteriorPoint(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)
        result = minimizer.minimize(x0, tol=tol, max_iter=max_iter)
        
        # Plot results
        minimizer.plot_feasible_region_and_path(self.plot_ranges[0], self.plot_ranges[1])
        minimizer.plot_objective_values()
        
        # Print final values
        minimizer.print_final_values(result['x'])
        
        print("Final result:", result)
        print("Final x:", result['x'])
        print("Final f(x):", result['f'])

        # Assertions
        self.assertTrue(result['success'], "Interior point method failed to converge")
        self.assertTrue(np.isfinite(result['f']), "Interior point method produced non-finite result")
        self.assertTrue(np.allclose(result['x'], np.array([0, 0, 1]), atol=tol), "Interior point method did not converge to the correct solution")

if __name__ == '__main__':
    unittest.main()