def parse_and_fit_arcfit_model(spec_string):
    """
    Parse and fit an ARCFIT model specification using non-linear least squares.

    Utility: Parses ARCFIT format, extracts model type, parameters, and data points,
    then fits the specified model to the data using scipy's curve fitting capabilities.
    Supports exp_decay, power_law, and logistic models.

    Args:
        spec_string (str): ARCFIT specification in format 
                          MODEL:<name>;PARAMS:<key>=<val_or_?>,...;DATA:<x1>,<y1>|<x2>,<y2>|...

    Returns:
        dict: JSON object mapping parameter names to fitted values (rounded to 6 decimals)
    """
    import re
    import math

    # Simple optimization function for curve fitting
    def minimize_residuals(func, x_data, y_data, param_names, initial_guess, bounds=None):
        """Simple gradient-free optimization using coordinate descent"""
        params = list(initial_guess)
        learning_rate = 0.1
        tolerance = 1e-8
        max_iterations = 10000

        def compute_residual_sum():
            try:
                y_pred = [func(x, *params) for x in x_data]
                return sum((y_actual - y_pred_val)**2 for y_actual, y_pred_val in zip(y_data, y_pred))
            except (OverflowError, ValueError, ZeroDivisionError):
                return float('inf')

        best_residual = compute_residual_sum()

        for iteration in range(max_iterations):
            improved = False

            for i in range(len(params)):
                original_param = params[i]

                # Try small steps in both directions
                for direction in [1, -1]:
                    step_size = learning_rate * direction
                    params[i] = original_param + step_size

                    # Apply bounds if provided
                    if bounds and bounds[i]:
                        params[i] = max(bounds[i][0], min(bounds[i][1], params[i]))

                    residual = compute_residual_sum()

                    if residual < best_residual:
                        best_residual = residual
                        improved = True
                        break
                    else:
                        params[i] = original_param

            if not improved:
                learning_rate *= 0.95
                if learning_rate < tolerance:
                    break

        return dict(zip(param_names, params))

    # Parse the specification
    parts = spec_string.split(';')

    # Extract model name
    model_match = re.match(r'MODEL:(\w+)', parts[0])
    if not model_match:
        raise ValueError("Invalid model specification")
    model_name = model_match.group(1)

    # Extract parameters
    params_match = re.match(r'PARAMS:(.*)', parts[1])
    if not params_match:
        raise ValueError("Invalid parameters specification")

    param_specs = {}
    for param_pair in params_match.group(1).split(','):
        key, val = param_pair.split('=')
        param_specs[key.strip()] = val.strip()

    # Extract data points
    data_match = re.match(r'DATA:(.*)', parts[2])
    if not data_match:
        raise ValueError("Invalid data specification")

    x_data = []
    y_data = []
    for point in data_match.group(1).split('|'):
        x_str, y_str = point.split(',')
        x_data.append(float(x_str))
        y_data.append(float(y_str))

    # Define model functions
    def exp_decay(x, a, b, c):
        return a * math.exp(-b * x) + c

    def power_law(x, a, b, c):
        if x <= 0 and b != int(b):
            return float('inf')
        return a * (x ** b) + c

    def logistic(x, L, k, x0):
        try:
            return L / (1 + math.exp(-k * (x - x0)))
        except OverflowError:
            return L if k * (x - x0) > 0 else 0

    # Select model function and set up parameters
    if model_name == 'exp_decay':
        model_func = exp_decay
        param_names = ['a', 'b', 'c']
        # Initial guess based on data characteristics
        y_max, y_min = max(y_data), min(y_data)
        initial_guess = [y_max - y_min, 0.5, y_min]
        bounds = [(0.01, 100), (0.01, 10), (-100, 100)]

    elif model_name == 'power_law':
        model_func = power_law
        param_names = ['a', 'b', 'c']
        initial_guess = [1.0, 0.5, min(y_data)]
        bounds = [(-100, 100), (-10, 10), (-100, 100)]

    elif model_name == 'logistic':
        model_func = logistic
        param_names = ['L', 'k', 'x0']
        y_max, y_min = max(y_data), min(y_data)
        x_mid = (max(x_data) + min(x_data)) / 2
        initial_guess = [y_max, 1.0, x_mid]
        bounds = [(0.01, 1000), (0.01, 10), (min(x_data), max(x_data))]

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Identify free parameters and create parameter mapping
    free_param_names = []
    fixed_params = {}
    
    for param_name in param_names:
        if param_name in param_specs:
            if param_specs[param_name] == '?':
                free_param_names.append(param_name)
            else:
                fixed_params[param_name] = float(param_specs[param_name])

    # Create wrapper function that handles fixed parameters
    def model_wrapper(x, *free_params):
        # Combine free and fixed parameters
        full_params = {}
        free_param_dict = dict(zip(free_param_names, free_params))
        full_params.update(fixed_params)
        full_params.update(free_param_dict)
        
        # Call model function with parameters in correct order
        ordered_params = [full_params[name] for name in param_names]
        return model_func(x, *ordered_params)

    # Get initial guess and bounds for free parameters only
    free_initial_guess = []
    free_bounds = []
    for i, param_name in enumerate(param_names):
        if param_name in free_param_names:
            free_initial_guess.append(initial_guess[i])
            free_bounds.append(bounds[i])

    # Fit the model
    fitted_params = minimize_residuals(
        model_wrapper, 
        x_data, 
        y_data, 
        free_param_names, 
        free_initial_guess, 
        free_bounds
    )

    # Combine fitted and fixed parameters
    result = {}
    result.update(fixed_params)
    result.update(fitted_params)

    # Round to 6 decimal places
    for key in result:
        result[key] = round(result[key], 6)

    return result