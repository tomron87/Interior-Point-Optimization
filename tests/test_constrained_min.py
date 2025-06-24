import unittest
import numpy as np
import sys
import os
import pandas as pd
from datetime import datetime
from src.constrained_min import InteriorPoint
from tests.examples import quadratic, quadratic_ineq1, quadratic_ineq2, quadratic_ineq3, linear, linear_ineq1, linear_ineq2, linear_ineq3, linear_ineq4
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_3d_path_figure(df: pd.DataFrame, title: str = "Central path"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # triangle
    verts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ax.add_collection3d(
        Poly3DCollection([verts], alpha=0.25, facecolor="tab:blue", edgecolor="k")
    )

    # central path
    ax.plot(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], "-o", lw=1.5, ms=5, label="central path")
    ax.plot(
        df.iloc[-1, 0],
        df.iloc[-1, 1],
        df.iloc[-1, 2],
        "r*",
        ms=12,
        label="final solution",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    return fig

def plot_feasible_region_and_path_2d(df: pd.DataFrame, ineq_constraints, title="Central path"):
    import numpy as np
    import matplotlib.pyplot as plt

    grid_res = 400
    padding = 0.1
    x_lo, x_hi = df.iloc[:, 0].min(), df.iloc[:, 0].max()
    y_lo, y_hi = df.iloc[:, 1].min(), df.iloc[:, 1].max()
    x_pad = (x_hi - x_lo) * padding or 1.0
    y_pad = (y_hi - y_lo) * padding or 1.0
    x_lo, x_hi = x_lo - x_pad, x_hi + x_pad
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad

    xx, yy = np.meshgrid(
        np.linspace(x_lo, x_hi, grid_res),
        np.linspace(y_lo, y_hi, grid_res),
    )

    # Stack grid to shape (N*N, 2)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # shape (N*N, 2)

    feas = np.ones(grid_pts.shape[0], dtype=bool)
    for g in ineq_constraints:
        feas &= np.array([g(pt)[0] <= 0 for pt in grid_pts])

    # Reshape feas to (grid_res, grid_res)
    feas = feas.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        feas.astype(float),
        extent=[x_lo, x_hi, y_lo, y_hi],
        origin="lower",
        cmap="Greys",
        alpha=0.3,
        aspect="auto",
    )
    ax.plot(df.iloc[:, 0], df.iloc[:, 1], "-o", ms=5, lw=1.5, label="central path")
    ax.plot(df.iloc[-1, 0], df.iloc[-1, 1], "r*", ms=12, label="final solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, ls="--", lw=0.5)
    return fig
def plot_objective_values(objective_values):
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = range(len(objective_values))
    ax.plot(iterations, objective_values, 'b-', linewidth=2)
    ax.set_title('Objective Value vs Iteration')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.grid(True)
    return fig
class TestConstrainedMin(unittest.TestCase):
    def setUp(self):
        # Create logs directory if it doesn't exist
        self.logs_dir = 'logs'
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        # Create plots directory if it doesn't exist
        self.plots_dir = 'plots'
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            
        # Set up logging to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.logs_dir, f'test_results_{self._testMethodName}_{timestamp}.log')
        self.original_stdout = sys.stdout
        self.log = open(self.log_file, 'w')
        sys.stdout = self.log
        
        # Test setup
        self.functions = [quadratic, linear]
        self.ineq_constraints = {'qp': [quadratic_ineq1, quadratic_ineq2, quadratic_ineq3], 
                                 'lp': [linear_ineq1, linear_ineq2, linear_ineq3, linear_ineq4]}
        self.eq_contraints_mat = {'qp': np.array([[1, 1, 1]]), 'lp': None}
        self.eq_constraints_rhs = {'qp': np.array([1]), 'lp': None}
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
        x0 = self.x0[0]
        func = self.functions[0]
        ineq_constraints = self.ineq_constraints['qp']
        eq_constraints_mat = self.eq_contraints_mat['qp']
        eq_constraints_rhs = self.eq_constraints_rhs['qp']

        print("Initial inequality constraint values (QP):")
        for i, g in enumerate(ineq_constraints):
            print(f"g{i}(x0) = {g(x0)[0]}")
        
        minimizer = InteriorPoint(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)
        result = minimizer.minimize(x0, tol=tol)
        
        # Plot results and save to file
        df = pd.DataFrame(result['path'][0])
        fig1 = draw_3d_path_figure(df, title=f"Quadratic Problem - Feasible Region")
        fig1.savefig(os.path.join(self.plots_dir, f"{self._testMethodName}_feasible_region.png"))
        fig2 = plot_objective_values(result['path'][1])
        fig2.savefig(os.path.join(self.plots_dir, f"{self._testMethodName}_objective_values.png"))
        
        # Print final values
        minimizer.print_final_values(result['x'])

        print("Final result:", result)
        print("Final x:", result['x'])
        print("Final f(x):", result['f'])
        
        # Assertions
        self.assertTrue(result['success'], "Interior point method failed to converge")
        self.assertTrue(np.isfinite(result['f']), "Interior point method produced non-finite result")

    def test_lp(self):
        print("\nTesting Linear Programming Problem:")
        tol = 1e-12
        x0 = self.x0[1]
        func = self.functions[1]
        ineq_constraints = self.ineq_constraints['lp']
        eq_constraints_mat = self.eq_contraints_mat['lp']
        eq_constraints_rhs = self.eq_constraints_rhs['lp']

        print("Initial inequality constraint values (LP):")
        for i, g in enumerate(ineq_constraints):
            print(f"g{i}(x0) = {g(x0)[0]}")
        
        minimizer = InteriorPoint(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)
        result = minimizer.minimize(x0, tol=tol)
        
        # Plot results and save to file
        df = pd.DataFrame(result['path'][0])
        fig1 = plot_feasible_region_and_path_2d(df, ineq_constraints, title=f"Linear Problem - Feasible Region")    
        fig1.savefig(os.path.join(self.plots_dir, f"{self._testMethodName}_feasible_region.png"))
        fig2 = plot_objective_values(result['path'][1])
        fig2.savefig(os.path.join(self.plots_dir, f"{self._testMethodName}_objective_values.png"))
        
        # Print final values
        minimizer.print_final_values(result['x'])
        
        print("Final result:", result)
        print("Final x:", result['x'])
        print("Final f(x):", result['f'])

        # Assertions
        self.assertTrue(result['success'], "Interior point method failed to converge")
        self.assertTrue(np.isfinite(result['f']), "Interior point method produced non-finite result")

if __name__ == '__main__':
    unittest.main()