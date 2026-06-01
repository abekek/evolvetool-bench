def parse_and_fit_arcfit_model(arcfit_spec):
    """
    Parse and fit an ARCFIT model specification to data using non-linear least squares.
    
    Utility: Parses ARCFIT format strings and fits supported curve models (exp_decay, power_law, logistic)
    to provided data points. Uses scipy.optimize for non-linear fitting with automatic parameter estimation.
    
    Args:
        arcfit_spec (str): ARCFIT specification string in format 
                          "MODEL:<name>;PARAMS:<key>=<val_or_?>,...;DATA:<x1>,<y1>|<x2>,<y2>|..."
    
    Returns:
        dict: JSON object mapping parameter names to fitted values rounded to 6 decimal places
    """
    import json
    import math
    from scipy.optimize import curve_fit
    import numpy as np
    
    # Parse the ARCFIT specification
    parts = arcfit_spec.split(';')
    model_part = parts[0]
    params_part = parts[1] 
    data_part = parts[2]
    
    # Extract model name
    model_name = model_part.split(':')[1]
    
    # Extract parameters
    params_str = params_part.split(':')[1]
    param_pairs = params_str.split(',')
    param_info = {}
    for pair in param_pairs:
        key, val = pair.split('=')
        if val == '?':
            param_info[key] = None  # Free parameter
        else:
            param_info[key] = float(val)  # Fixed parameter
    
    # Extract data points
    data_str = data_part.split(':')[1]
    data_points = data_str.split('|')
    x_data = []
    y_data = []
    for point in data_points:
        x_val, y_val = point.split(',')
        x_data.append(float(x_val))
        y_data.append(float(y_val))
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Define model functions
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    def power_law(x, a, b, c):
        return a * np.power(x, b) + c
    
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
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
    
    # Prepare initial guesses for free parameters
    if model_name == 'exp_decay':
        # For exp_decay: y = a * exp(-b * x) + c
        # Estimate: c ~ min(y), a ~ max(y) - min(y), b ~ 1
        initial_guess = [max(y_data) - min(y_data), 1.0, min(y_data)]
    elif model_name == 'power_law':
        initial_guess = [1.0, 1.0, 0.0]
    else:  # logistic
        initial_guess = [max(y_data), 1.0, np.mean(x_data)]
    
    # Handle fixed parameters by creating wrapper function
    free_params = [name for name in param_names if param_info.get(name) is None]
    fixed_params = {name: val for name, val in param_info.items() if val is not None}
    
    if len(fixed_params) == 0:
        # All parameters are free
        fitted_params, _ = curve_fit(model_func, x_data, y_data, p0=initial_guess)
        result = {param_names[i]: round(fitted_params[i], 6) for i in range(len(param_names))}
    else:
        # Some parameters are fixed - create wrapper function
        def wrapper_func(x, *free_vals):
            full_params = {}
            free_idx = 0
            for param in param_names:
                if param in fixed_params:
                    full_params[param] = fixed_params[param]
                else:
                    full_params[param] = free_vals[free_idx]
                    free_idx += 1
            
            if model_name == 'exp_decay':
                return model_func(x, full_params['a'], full_params['b'], full_params['c'])
            elif model_name == 'power_law':
                return model_func(x, full_params['a'], full_params['b'], full_params['c'])
            else:  # logistic
                return model_func(x, full_params['L'], full_params['k'], full_params['x0'])
        
        free_initial_guess = [initial_guess[i] for i, name in enumerate(param_names) if name in free_params]
        fitted_free_params, _ = curve_fit(wrapper_func, x_data, y_data, p0=free_initial_guess)
        
        result = {}
        free_idx = 0
        for param in param_names:
            if param in fixed_params:
                result[param] = round(fixed_params[param], 6)
            else:
                result[param] = round(fitted_free_params[free_idx], 6)
                free_idx += 1
    
    return result