def fit_arcfit_exponential_decay(spec_string, eval_points):
    """
    Fits an exponential decay model to data and evaluates predictions with statistics.
    
    Utility: Parses ARCFIT specification string, fits exponential decay model (y = a * exp(-b * x) + c)
    using least squares optimization, evaluates model at given points, and computes statistics.
    
    Args:
        spec_string (str): ARCFIT specification with MODEL, PARAMS, and DATA sections
        eval_points (list): List of x values where to evaluate the fitted model
        
    Returns:
        dict: JSON object with 'fitted_params' (a,b,c), 'predictions' (y values), 
              and 'stats' (mean, median, std of predictions)
    """
    import math
    import json
    
    # Parse the specification string
    sections = spec_string.split(';')
    model_type = sections[0].split(':')[1]
    data_section = sections[2].split(':')[1]
    
    # Parse data points
    data_pairs = data_section.split('|')
    x_data = []
    y_data = []
    for pair in data_pairs:
        x_val, y_val = map(float, pair.split(','))
        x_data.append(x_val)
        y_data.append(y_val)
    
    # Fit exponential decay model: y = a * exp(-b * x) + c
    # Use simple grid search optimization for robustness
    best_error = float('inf')
    best_params = None
    
    # Grid search ranges
    a_range = [i * 0.5 for i in range(1, 21)]  # 0.5 to 10
    b_range = [i * 0.1 for i in range(1, 31)]  # 0.1 to 3.0
    c_range = [i * 0.1 for i in range(0, 21)]  # 0 to 2.0
    
    for a in a_range:
        for b in b_range:
            for c in c_range:
                error = 0
                for i in range(len(x_data)):
                    predicted = a * math.exp(-b * x_data[i]) + c
                    error += (y_data[i] - predicted) ** 2
                
                if error < best_error:
                    best_error = error
                    best_params = {'a': a, 'b': b, 'c': c}
    
    # Fine-tune around best parameters
    a, b, c = best_params['a'], best_params['b'], best_params['c']
    step_size = 0.01
    
    for _ in range(50):  # Limited iterations for fine-tuning
        gradients = {'a': 0, 'b': 0, 'c': 0}
        
        for i in range(len(x_data)):
            exp_term = math.exp(-b * x_data[i])
            predicted = a * exp_term + c
            residual = predicted - y_data[i]
            
            gradients['a'] += 2 * residual * exp_term
            gradients['b'] += 2 * residual * a * (-x_data[i]) * exp_term
            gradients['c'] += 2 * residual
        
        # Update parameters
        a -= step_size * gradients['a'] / len(x_data)
        b -= step_size * gradients['b'] / len(x_data)
        c -= step_size * gradients['c'] / len(x_data)
        
        # Keep parameters in reasonable bounds
        a = max(0.1, min(20, a))
        b = max(0.01, min(5, b))
        c = max(0, min(5, c))
    
    fitted_params = {'a': round(a, 6), 'b': round(b, 6), 'c': round(c, 6)}
    
    # Evaluate model at specified points
    predictions = []
    for x in eval_points:
        y_pred = a * math.exp(-b * x) + c
        predictions.append(round(y_pred, 6))
    
    # Compute statistics
    n = len(predictions)
    mean_val = sum(predictions) / n
    
    sorted_preds = sorted(predictions)
    if n % 2 == 0:
        median_val = (sorted_preds[n//2 - 1] + sorted_preds[n//2]) / 2
    else:
        median_val = sorted_preds[n//2]
    
    variance = sum((p - mean_val) ** 2 for p in predictions) / n
    std_val = math.sqrt(variance)
    
    stats = {
        'mean': round(mean_val, 6),
        'median': round(median_val, 6),
        'std': round(std_val, 6)
    }
    
    result = {
        'fitted_params': fitted_params,
        'predictions': predictions,
        'stats': stats
    }
    
    return json.dumps(result, indent=2)