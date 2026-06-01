def solve_arcopt_linear_program(spec):
    """
    Solve an ARCOPT linear programming problem using scipy's linear programming solver.
    
    Utility: Parses ARCOPT specification format and solves the linear optimization problem
    to find the minimum value of the objective function subject to constraints.
    
    Args:
        spec (str): ARCOPT specification string in format 
                   "ARCOPT:v1;VARS:...;OBJ:linear:...;CONSTRS:...;BOUNDS:..."
    
    Returns:
        dict: Dictionary containing "minimum" (float) and "at" (dict of variable values),
              both rounded to 6 decimal places
    """
    import re
    from scipy.optimize import linprog
    import numpy as np
    
    # Parse the specification
    parts = spec.split(';')
    
    # Extract variables
    vars_part = next(p for p in parts if p.startswith('VARS:'))
    variables = [v.strip() for v in vars_part[5:].split(',')]
    n_vars = len(variables)
    
    # Extract objective coefficients
    obj_part = next(p for p in parts if p.startswith('OBJ:linear:'))
    obj_expr = obj_part[11:]
    
    # Parse objective coefficients
    c = [0.0] * n_vars
    for i, var in enumerate(variables):
        pattern = rf'([+-]?\d*\.?\d*)\*?{var}'
        matches = re.findall(pattern, obj_expr)
        if matches:
            coef = matches[0]
            if coef == '' or coef == '+':
                c[i] = 1.0
            elif coef == '-':
                c[i] = -1.0
            else:
                c[i] = float(coef)
    
    # Extract constraints
    constrs_part = next(p for p in parts if p.startswith('CONSTRS:'))
    constraints = [c.strip() for c in constrs_part[9:].split('|')]
    
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    for constraint in constraints:
        if '>=' in constraint:
            left, right = constraint.split('>=')
            # Convert >= to <= by negating
            coeffs = [0.0] * n_vars
            for i, var in enumerate(variables):
                pattern = rf'([+-]?\d*\.?\d*)\*?{var}'
                matches = re.findall(pattern, left)
                if matches:
                    coef = matches[0]
                    if coef == '' or coef == '+':
                        coeffs[i] = 1.0
                    elif coef == '-':
                        coeffs[i] = -1.0
                    else:
                        coeffs[i] = float(coef)
            A_ub.append([-c for c in coeffs])
            b_ub.append(-float(right.strip()))
        elif '<=' in constraint:
            left, right = constraint.split('<=')
            coeffs = [0.0] * n_vars
            for i, var in enumerate(variables):
                pattern = rf'([+-]?\d*\.?\d*)\*?{var}'
                matches = re.findall(pattern, left)
                if matches:
                    coef = matches[0]
                    if coef == '' or coef == '+':
                        coeffs[i] = 1.0
                    elif coef == '-':
                        coeffs[i] = -1.0
                    else:
                        coeffs[i] = float(coef)
            A_ub.append(coeffs)
            b_ub.append(float(right.strip()))
    
    # Extract bounds
    bounds_part = next(p for p in parts if p.startswith('BOUNDS:'))
    bounds_specs = [b.strip() for b in bounds_part[7:].split('|')]
    
    bounds = []
    for bound_spec in bounds_specs:
        var, bound_range = bound_spec.split(':')
        bound_range = bound_range.strip('[]')
        lower, upper = bound_range.split(',')
        
        lower_val = 0.0 if lower.strip() == '0' else float(lower.strip())
        upper_val = None if upper.strip() == '+inf' else float(upper.strip())
        
        bounds.append((lower_val, upper_val))
    
    # Solve the linear program
    result = linprog(c, A_ub=A_ub if A_ub else None, b_ub=b_ub if b_ub else None, 
                    A_eq=A_eq if A_eq else None, b_eq=b_eq if b_eq else None,
                    bounds=bounds, method='highs')
    
    if result.success:
        minimum = round(result.fun, 6)
        solution_dict = {variables[i]: round(result.x[i], 6) for i in range(n_vars)}
        return {"minimum": minimum, "at": solution_dict}
    else:
        return {"error": "No feasible solution found"}