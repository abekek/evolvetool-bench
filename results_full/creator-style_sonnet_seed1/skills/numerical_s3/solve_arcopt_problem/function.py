def solve_arcopt_problem(spec):
    """
    Solves an ARCOPT optimization problem using scipy's linear or quadratic solvers.
    
    Utility: Parses ARCOPT specification string and solves the optimization problem,
    returning the minimum value and optimal variable values rounded to 6 decimal places.
    
    Args:
        spec (str): ARCOPT specification string containing variables, objective, constraints, and bounds
        
    Returns:
        dict: JSON object with "minimum" (float) and "at" (dict of variable values)
    """
    import numpy as np
    from scipy.optimize import linprog, minimize
    import re
    
    # Parse the specification
    parts = spec.split(';')
    
    # Extract variables
    vars_part = next(p for p in parts if p.startswith('VARS:'))
    variables = [v.strip() for v in vars_part[5:].split(',')]
    n_vars = len(variables)
    
    # Extract objective
    obj_part = next(p for p in parts if p.startswith('OBJ:'))
    obj_content = obj_part[4:]
    is_linear = obj_content.startswith('linear:')
    obj_expr = obj_content.split(':', 1)[1]
    
    # Parse objective coefficients
    c = [0] * n_vars
    for i, var in enumerate(variables):
        pattern = r'([+-]?\d*\.?\d*)\*' + var
        matches = re.findall(pattern, obj_expr)
        if matches:
            coeff = matches[0]
            if coeff == '' or coeff == '+':
                c[i] = 1
            elif coeff == '-':
                c[i] = -1
            else:
                c[i] = float(coeff)
    
    # Extract constraints
    constrs_part = next(p for p in parts if p.startswith('CONSTRS:'))
    constraints = constrs_part[8:].split('|')
    
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    for constr in constraints:
        # Parse constraint coefficients and operator
        if '>=' in constr:
            left, right = constr.split('>=')
            # Convert >= to <= by negating
            coeffs = [-1] * n_vars
            for i, var in enumerate(variables):
                pattern = r'([+-]?\d*\.?\d*)\*' + var
                matches = re.findall(pattern, left)
                if matches:
                    coeff = matches[0]
                    if coeff == '' or coeff == '+':
                        coeffs[i] = -1
                    elif coeff == '-':
                        coeffs[i] = 1
                    else:
                        coeffs[i] = -float(coeff)
            A_ub.append(coeffs)
            b_ub.append(-float(right.strip()))
            
        elif '<=' in constr:
            left, right = constr.split('<=')
            coeffs = [0] * n_vars
            for i, var in enumerate(variables):
                pattern = r'([+-]?\d*\.?\d*)\*' + var
                matches = re.findall(pattern, left)
                if matches:
                    coeff = matches[0]
                    if coeff == '' or coeff == '+':
                        coeffs[i] = 1
                    elif coeff == '-':
                        coeffs[i] = -1
                    else:
                        coeffs[i] = float(coeff)
            A_ub.append(coeffs)
            b_ub.append(float(right.strip()))
    
    # Extract bounds
    bounds_part = next(p for p in parts if p.startswith('BOUNDS:'))
    bounds_specs = bounds_part[7:].split('|')
    bounds = []
    
    for bound_spec in bounds_specs:
        var_name, bound_range = bound_spec.split(':')
        bound_range = bound_range.strip('[]')
        lower, upper = bound_range.split(',')
        lower = 0 if lower == '0' else (None if lower == '-inf' else float(lower))
        upper = None if upper == '+inf' else float(upper)
        bounds.append((lower, upper))
    
    # Solve the optimization problem
    if is_linear:
        result = linprog(c, A_ub=A_ub if A_ub else None, b_ub=b_ub if b_ub else None,
                        A_eq=A_eq if A_eq else None, b_eq=b_eq if b_eq else None,
                        bounds=bounds, method='highs')
        
        minimum = round(result.fun, 6)
        solution = {var: round(val, 6) for var, val in zip(variables, result.x)}
    else:
        # For quadratic objectives, use minimize with SLSQP
        def objective(x):
            return sum(c[i] * x[i] for i in range(len(x)))
        
        constraints = []
        if A_ub:
            for i in range(len(A_ub)):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: b_ub[i] - sum(A_ub[i][j] * x[j] for j in range(len(x)))
                })
        
        x0 = [1] * n_vars  # Initial guess
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        minimum = round(result.fun, 6)
        solution = {var: round(val, 6) for var, val in zip(variables, result.x)}
    
    return {"minimum": minimum, "at": solution}