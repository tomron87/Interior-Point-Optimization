# Interior Point Optimization

This project implements an interior point method for solving constrained optimization problems in Python, using only numpy for core optimization routines. It supports both inequality and equality constraints, and includes visualization tools for feasible regions and optimization paths.

## Features
- Log-barrier interior point method for constrained minimization
- Support for both 2D and 3D problems
- Visualization of feasible regions and central path
- Unit tests for quadratic and linear programming examples

## Requirements
- Python 3.8+
- numpy
- pandas
- matplotlib

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run unit tests:
   ```bash
   python -m unittest discover tests
   ```
3. Use the `InteriorPoint` class in your own scripts for constrained optimization problems.

## File Structure
- `src/` - Source code for optimization algorithms
- `tests/` - Unit tests and example problems

## Example
See `tests/test_constrained_min.py` for example usage and test cases. 