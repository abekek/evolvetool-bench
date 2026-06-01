def solve_arcopt_linear_problem(spec_string):
    """
    Solves an ARCOPT linear programming optimization problem by parsing the specification
    and using scipy's linear programming solver.

    Args:
        spec_string (str): ARCOPT specification string containing variables, objective, 
                          constraints, and bounds in the format:
                          ARCOPT:v1;VARS:...;OBJ:linear:...;CONSTRS:...;BOUNDS:...

    Returns:
        dict: Dictionary with 'minimum' (float) and 'at' (dict) keys containing
              the optimal objective value and variable values, rounded to 6 decimal places
    """
    import re
    from scipy.optimize import linprog
    import numpy as np

    # Parse the specification
    parts = spec_string.split(';')

    # Extract variables
    vars_part = next(p for p in parts if p.startswith('VARS:'))
    variables = [v.strip() for v in vars_part.split(':')[1].split(',')]
    n_vars = len(variables)

    # Extract objective coefficients
    obj_part = next(p for p in parts if p.startswith('OBJ:'))
    obj_expr = obj_part.split(':', 2)[2]
    obj_coeffs = [0.0] * n_vars

    for i, var in enumerate(variables):
        # Improved pattern to match coefficients
        pattern = r'([+-]?\d*\.?\d*)\*?' + re.escape(var) + r'(?![a-zA-Z0-9_])'
        matches = re.findall(pattern, obj_expr)
        if matches:
            coeff = matches[0]
            if coeff == '' or coeff == '+':
                obj_coeffs[i] = 1.0
            elif coeff == '-':
                obj_coeffs[i] = -1.0
            else:
                obj_coeffs[i] = float(coeff)

    # Extract constraints
    constrs_part = next(p for p in parts if p.startswith('CONSTRS:'))
    constraint_strs = constrs_part.split(':')[1].split('|')

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    for constr in constraint_strs:
        # Parse constraint like "1*x1+2*x2<=10" or "1*x1+1*x2+1*x3>=6"
        if '>=' in constr:
            lhs, rhs = constr.split('>=')
            inequality = '>='
        elif '<=' in constr:
            lhs, rhs = constr.split('<=')
            inequality = '<='
        elif '=' in constr:
            lhs, rhs = constr.split('=')
            inequality = '='

        # Extract coefficients
        coeffs = [0.0] * n_vars
        for i, var in enumerate(variables):
            pattern = r'([+-]?\d*\.?\d*)\*?' + re.escape(var) + r'(?![a-zA-Z0-9_])'
            matches = re.findall(pattern, lhs)
            if matches:
                coeff = matches[0]
                if coeff == '' or coeff == '+':
                    coeffs[i] = 1.0
                elif coeff == '-':
                    coeffs[i] = -1.0
                else:
                    coeffs[i] = float(coeff)

        rhs_val = float(rhs.strip())

        if inequality == '<=':
            A_ub.append(coeffs)
            b_ub.append(rhs_val)
        elif inequality == '>=':
            # Convert >= to <= by negating both sides
            A_ub.append([-c for c in coeffs])
            b_ub.append(-rhs_val)
        elif inequality == '=':
            A_eq.append(coeffs)
            b_eq.append(rhs_val)

    # Extract bounds - Fix the parsing here
    bounds_part = next(p for p in parts if p.startswith('BOUNDS:'))
    bounds_strs = bounds_part.split(':', 1)[1].split('|')

    bounds = []
    for bound_str in bounds_strs:
        # Split on the first colon to separate variable name from bounds
        colon_idx = bound_str.find(':')
        var_name = bound_str[:colon_idx]
        bound_range = bound_str[colon_idx+1:]
        
        # Remove brackets and split on comma
        bound_range = bound_range.strip('[]')
        lower, upper = bound_range.split(',')

        if lower.strip() == '-inf':
            lower_val = None
        elif lower.strip() == '+inf':
            lower_val = float('inf')
        else:
            lower_val = float(lower.strip())

        if upper.strip() == '+inf':
            upper_val = None
        elif upper.strip() == '-inf':
            upper_val = float('-inf')
        else:
            upper_val = float(upper.strip())

        bounds.append((lower_val, upper_val))

    # Prepare inputs for scipy.optimize.linprog
    c = obj_coeffs  # Objective coefficients (linprog minimizes by default)

    A_ub_array = np.array(A_ub) if A_ub else None
    b_ub_array = np.array(b_ub) if b_ub else None

    A_eq_array = np.array(A_eq) if A_eq else None
    b_eq_array = np.array(b_eq) if b_eq else None

    # Solve the linear program
    result = linprog(c, A_ub=A_ub_array, b_ub=b_ub_array, 
                     A_eq=A_eq_array, b_eq=b_eq_array, 
                     bounds=bounds, method='highs')

    if result.success:
        minimum = round(result.fun, 6)
        at = {var: round(result.x[i], 6) for i, var in enumerate(variables)}
        return {"minimum": minimum, "at": at}
    else:
        raise ValueError(f"Optimization failed: {result.message}")