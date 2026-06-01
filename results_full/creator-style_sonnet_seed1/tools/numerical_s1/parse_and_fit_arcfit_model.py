def parse_and_fit_arcfit_model(spec_string):
    """
    Parse and fit an ARCFIT model specification using non-linear least squares.
    
    Utility: Parses ARCFIT format strings and fits supported models (exp_decay, power_law, logistic)
    to provided data points using scipy's curve_fit for non-linear parameter estimation.
    
    Args:
        spec_string (str): ARCFIT format string containing MODEL, PARAMS, and DATA sections
    
    Returns:
        dict: Parameter names mapped to fitted values rounded to 6 decimal places
    """
    import re
    import math
    from typing import List, Tuple, Dict, Any, Callable
    
    # Parse the specification string
    parts = spec_string.split(';')
    model_part = next(p for p in parts if p.startswith('MODEL:'))
    params_part = next(p for p in parts if p.startswith('PARAMS:'))
    data_part = next(p for p in parts if p.startswith('DATA:'))
    
    # Extract model name
    model_name = model_part.split(':')[1]
    
    # Extract parameters
    params_str = params_part.split(':', 1)[1]
    param_specs = {}
    for param in params_str.split(','):
        key, val = param.split('=')
        param_specs[key] = val
    
    # Extract data points
    data_str = data_part.split(':', 1)[1]
    data_points = []
    for point in data_str.split('|'):
        x, y = point.split(',')
        data_points.append((float(x), float(y)))
    
    x_data = [p[0] for p in data_points]
    y_data = [p[1] for p in data_points]
    
    # Define model functions
    def exp_decay(x, a, b, c):
        return a * math.exp(-b * x) + c
    
    def power_law(x, a, b, c):
        return a * (x ** b) + c
    
    def logistic(x, L, k, x0):
        return L / (1 + math.exp(-k * (x - x0)))
    
    # Select model function and parameter names
    if model_name == 'exp_decay':
        model_func = exp_decay
        param_names = ['a', 'b', 'c']
    elif model_name == 'power_law':
        model_func = power_law
        param_names = ['a', 'b', 'c']
    elif model_name == 'logistic':
        model_func = logistic
        param_names = ['L', 'k', 'x0']
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Simple non-linear least squares implementation using Gauss-Newton method
    def fit_model(x_data, y_data, model_func, param_names, param_specs):
        # Initialize parameters
        initial_params = []
        fixed_params = {}
        free_param_indices = []
        
        for i, param in enumerate(param_names):
            if param_specs[param] == '?':
                # Initial guess for free parameters
                if model_name == 'exp_decay':
                    if param == 'a': initial_params.append(max(y_data) - min(y_data))
                    elif param == 'b': initial_params.append(0.5)
                    elif param == 'c': initial_params.append(min(y_data))
                elif model_name == 'power_law':
                    if param == 'a': initial_params.append(1.0)
                    elif param == 'b': initial_params.append(1.0)
                    elif param == 'c': initial_params.append(0.0)
                elif model_name == 'logistic':
                    if param == 'L': initial_params.append(max(y_data))
                    elif param == 'k': initial_params.append(1.0)
                    elif param == 'x0': initial_params.append(sum(x_data) / len(x_data))
                free_param_indices.append(i)
            else:
                fixed_params[i] = float(param_specs[param])
        
        # Simple optimization using numerical gradients
        params = initial_params[:]
        learning_rate = 0.01
        
        for iteration in range(1000):
            # Calculate residuals
            residuals = []
            for x, y in zip(x_data, y_data):
                # Construct full parameter list
                full_params = [0] * len(param_names)
                free_idx = 0
                for i in range(len(param_names)):
                    if i in fixed_params:
                        full_params[i] = fixed_params[i]
                    else:
                        full_params[i] = params[free_idx]
                        free_idx += 1
                
                try:
                    pred = model_func(x, *full_params)
                    residuals.append(y - pred)
                except (OverflowError, ZeroDivisionError):
                    residuals.append(1e6)
            
            # Calculate cost
            cost = sum(r**2 for r in residuals) / len(residuals)
            
            # Calculate numerical gradients
            gradients = []
            h = 1e-8
            for i in range(len(params)):
                params_plus = params[:]
                params_plus[i] += h
                
                residuals_plus = []
                for x, y in zip(x_data, y_data):
                    full_params = [0] * len(param_names)
                    free_idx = 0
                    for j in range(len(param_names)):
                        if j in fixed_params:
                            full_params[j] = fixe