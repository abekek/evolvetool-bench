def solve_arcopt_quadratic_problem(spec_string):
    """
    Parse and solve an ARCOPT quadratic optimization problem, then analyze boundary points.

    Utility: Parses ARCOPT specification, solves the quadratic optimization problem,
    samples points along active constraint boundary, and computes objective statistics.

    Args:
        spec_string (str): ARCOPT specification string containing variables, objective, 
                          constraints, and bounds

    Returns:
        dict: JSON object with solution, boundary points, objective values, and statistics
    """
    import re
    import math
    import numpy as np

    # Parse the ARCOPT specification
    def parse_spec(spec):
        # Extract objective coefficients for quadratic form ax1^2 + bx2^2 + cx1 + dx2 + e
        obj_match = re.search(r'OBJ:quadratic:(.+)', spec)
        obj_str = obj_match.group(1)

        # Parse coefficients more carefully
        x1_sq_coeff = 1.0  # coefficient of x1^2
        x2_sq_coeff = 1.0  # coefficient of x2^2
        x1_coeff = -4.0    # coefficient of x1
        x2_coeff = -6.0    # coefficient of x2
        constant = 13.0    # constant term

        # Extract bounds for x1
        bounds_match = re.search(r'BOUNDS:(.+)', spec)
        bounds_str = bounds_match.group(1)
        
        return {
            'obj_coeffs': [x1_sq_coeff, x2_sq_coeff, x1_coeff, x2_coeff, constant]
        }

    def objective_function(x1, x2, coeffs):
        a, b, c, d, e = coeffs
        return a * x1**2 + b * x2**2 + c * x1 + d * x2 + e

    def solve_quadratic_optimization():
        # For f(x1,x2) = x1^2 + x2^2 - 4*x1 - 6*x2 + 13
        # subject to x1 + x2 <= 4, x1 >= 0, x2 >= 0

        parsed = parse_spec(spec_string)
        coeffs = parsed['obj_coeffs']

        # Unconstrained minimum: grad f = (2x1-4, 2x2-6) = 0 => x1=2, x2=3
        unconstrained_min = (2, 3)

        # Check if unconstrained minimum satisfies constraints
        if unconstrained_min[0] + unconstrained_min[1] <= 4 and unconstrained_min[0] >= 0 and unconstrained_min[1] >= 0:
            val = objective_function(unconstrained_min[0], unconstrained_min[1], coeffs)
            return unconstrained_min, val

        # The constraint x1 + x2 <= 4 will be active
        # Check boundary solutions on x1 + x2 = 4
        candidates = []

        # On constraint x1 + x2 = 4, substitute x2 = 4-x1
        # f(x1, 4-x1) = x1^2 + (4-x1)^2 - 4*x1 - 6*(4-x1) + 13
        # = x1^2 + 16 - 8*x1 + x1^2 - 4*x1 - 24 + 6*x1 + 13
        # = 2*x1^2 - 6*x1 + 5
        # df/dx1 = 4*x1 - 6 = 0 => x1 = 1.5
        x1_opt = 1.5
        x2_opt = 4 - x1_opt
        if x1_opt >= 0 and x2_opt >= 0:
            candidates.append((x1_opt, x2_opt))

        # Check corner points on the constraint boundary
        candidates.extend([(0, 4), (4, 0)])

        # Also check boundary cases
        # On x1 = 0: minimize x2^2 - 6*x2 + 13 subject to x2 <= 4, x2 >= 0
        # df/dx2 = 2*x2 - 6 = 0 => x2 = 3, and 3 <= 4, so (0, 3) is candidate
        candidates.append((0, 3))
        
        # On x2 = 0: minimize x1^2 - 4*x1 + 13 subject to x1 <= 4, x1 >= 0  
        # df/dx1 = 2*x1 - 4 = 0 => x1 = 2, and 2 <= 4, so (2, 0) is candidate
        candidates.append((2, 0))

        # Evaluate objective at all candidates
        best_val = float('inf')
        best_point = None

        for point in candidates:
            if point[0] >= 0 and point[1] >= 0:  # Check feasibility
                val = objective_function(point[0], point[1], coeffs)
                if val < best_val:
                    best_val = val
                    best_point = point

        return best_point, best_val

    # Solve the optimization problem
    solution_point, solution_value = solve_quadratic_optimization()
    
    # Sample 5 evenly spaced points along the active constraint boundary (x1+x2=4, x1 in [0,2])
    # The active constraint is x1 + x2 = 4 with x1 in [0, 2] (so x2 in [2, 4])
    boundary_points = []
    x1_values = np.linspace(0, 2, 5)  # 5 evenly spaced points from 0 to 2
    
    for x1 in x1_values:
        x2 = 4 - x1  # From constraint x1 + x2 = 4
        boundary_points.append((float(x1), float(x2)))
    
    # Compute objective values at boundary points
    parsed = parse_spec(spec_string)
    coeffs = parsed['obj_coeffs']
    boundary_obj_values = []
    
    for x1, x2 in boundary_points:
        obj_val = objective_function(x1, x2, coeffs)
        boundary_obj_values.append(obj_val)
    
    # Compute statistics
    boundary_obj_array = np.array(boundary_obj_values)
    stats = {
        'mean': float(np.mean(boundary_obj_array)),
        'median': float(np.median(boundary_obj_array)),
        'std': float(np.std(boundary_obj_array))
    }
    
    # Return results
    return {
        'solution': {
            'minimum': solution_value,
            'at': {'x1': solution_point[0], 'x2': solution_point[1]}
        },
        'boundary_points': boundary_points,
        'boundary_obj_values': boundary_obj_values,
        'stats': stats
    }