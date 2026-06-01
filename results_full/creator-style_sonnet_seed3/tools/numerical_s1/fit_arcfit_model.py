def fit_arcfit_model(spec_string):
    """
    Fit an ARCFIT model specification to data using non-linear optimization.
    
    Utility: Parses ARCFIT specification string, extracts model type, parameters, and data,
             then fits the specified model (currently supports power_law: y = a * x^b + c)
             using scipy's curve fitting functionality via least squares optimization.
    
    Args:
        spec_string (str): ARCFIT specification in format "MODEL:model_type;PARAMS:param_specs;DATA:x1,y1|x2,y2|..."
    
    Returns:
        str: JSON string containing fitted parameter values rounded to 6 decimal places
    """
    import json
    import math
    
    # Parse the specification string
    parts = spec_string.split(';')
    model_part = parts[0].split(':')[1]
    params_part = parts[1].split(':')[1]
    data_part = parts[2].split(':')[1]
    
    # Extract parameter names
    param_specs = params_part.split(',')
    param_names = [spec.split('=')[0] for spec in param_specs]
    
    # Parse data points
    data_pairs = data_part.split('|')
    x_data = []
    y_data = []
    for pair in data_pairs:
        x, y = pair.split(',')
        x_data.append(float(x))
        y_data.append(float(y))
    
    # Define power law function: y = a * x^b + c
    def power_law(x, a, b, c):
        return a * (x ** b) + c
    
    # Simple curve fitting using least squares optimization
    # We'll use a basic optimization approach since scipy isn't in standard library
    
    # Initial parameter guesses
    best_params = [1.0, 0.5, 0.0]  # [a, b, c]
    best_error = float('inf')
    
    # Grid search for reasonable starting points
    a_range = [0.1, 0.5, 1.0, 2.0, 5.0]
    b_range = [0.1, 0.3, 0.5, 0.7, 1.0]
    c_range = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    for a in a_range:
        for b in b_range:
            for c in c_range:
                try:
                    error = 0
                    for i in range(len(x_data)):
                        predicted = power_law(x_data[i], a, b, c)
                        error += (y_data[i] - predicted) ** 2
                    
                    if error < best_error:
                        best_error = error
                        best_params = [a, b, c]
                except:
                    continue
    
    # Refine with simple gradient descent-like optimization
    learning_rate = 0.001
    for iteration in range(1000):
        current_error = 0
        gradients = [0, 0, 0]
        
        for i in range(len(x_data)):
            try:
                x, y_actual = x_data[i], y_data[i]
                y_pred = power_law(x, best_params[0], best_params[1], best_params[2])
                error = y_actual - y_pred
                current_error += error ** 2
                
                # Approximate gradients
                h = 1e-8
                grad_a = (power_law(x, best_params[0] + h, best_params[1], best_params[2]) - y_pred) / h
                grad_b = (power_law(x, best_params[0], best_params[1] + h, best_params[2]) - y_pred) / h
                grad_c = (power_law(x, best_params[0], best_params[1], best_params[2] + h) - y_pred) / h
                
                gradients[0] += error * grad_a
                gradients[1] += error * grad_b
                gradients[2] += error * grad_c
            except:
                continue
        
        # Update parameters
        for j in range(3):
            best_params[j] += learning_rate * gradients[j]
        
        # Adaptive learning rate
        if iteration % 100 == 0:
            learning_rate *= 0.99
    
    # Create result dictionary
    result = {}
    for i, param_name in enumerate(param_names):
        result[param_name] = round(best_params[i], 6)
    
    return json.dumps(result)