def fit_arcfit_exponential_decay(spec_string):
    """
    Fits an exponential decay model to data and evaluates predictions with statistics.

    Utility: Parses ARCFIT specification, fits exp_decay model (y = a * exp(-b * x) + c),
             evaluates on x=[0,2,4,6,8,10], and computes prediction statistics.

    Args:
        spec_string (str): ARCFIT specification containing model type, parameters, and data

    Returns:
        dict: JSON object with 'fitted_params', 'predictions', and 'stats' keys
    """
    import re
    import math

    # Parse the specification string
    parts = spec_string.split(';')

    # Extract model type
    model_match = re.search(r'MODEL:(\w+)', parts[0])
    model_type = model_match.group(1) if model_match else None

    # Extract data points
    data_match = re.search(r'DATA:(.+)', parts[2])
    data_str = data_match.group(1) if data_match else ""

    # Parse data points
    x_data, y_data = [], []
    pairs = data_str.split('|')
    for pair in pairs:
        x_val, y_val = map(float, pair.split(','))
        x_data.append(x_val)
        y_data.append(y_val)

    # Fit exponential decay model: y = a * exp(-b * x) + c
    def exp_decay_model(x, a, b, c):
        return a * math.exp(-b * x) + c

    def fit_exp_decay(x_vals, y_vals):
        # Use Levenberg-Marquardt-like approach with multiple initial guesses
        best_params = None
        best_error = float('inf')
        
        # Try different initial parameter combinations
        c_candidates = [min(y_vals), min(y_vals) * 0.5, 0, min(y_vals) * 1.5]
        
        for c_init in c_candidates:
            try:
                # Estimate a and b given c
                # Transform: ln(y - c) = ln(a) - b*x
                valid_points = []
                for i in range(len(y_vals)):
                    if y_vals[i] > c_init + 1e-10:
                        valid_points.append((x_vals[i], y_vals[i]))
                
                if len(valid_points) < 3:
                    continue
                
                x_valid = [p[0] for p in valid_points]
                y_valid = [p[1] for p in valid_points]
                y_ln = [math.log(y - c_init) for y in y_valid]
                
                # Linear regression: y_ln = ln(a) - b*x
                n = len(x_valid)
                sum_x = sum(x_valid)
                sum_y_ln = sum(y_ln)
                sum_xy = sum(x_valid[i] * y_ln[i] for i in range(n))
                sum_x2 = sum(x * x for x in x_valid)
                
                # Solve for b (slope) and ln(a) (intercept)
                denom = n * sum_x2 - sum_x * sum_x
                if abs(denom) < 1e-10:
                    continue
                    
                b = -(n * sum_xy - sum_x * sum_y_ln) / denom  # negative because we want decay
                ln_a = (sum_y_ln + b * sum_x) / n
                a = math.exp(ln_a)
                
                # Ensure b is positive for decay
                b = abs(b)
                
                # Calculate fit error
                error = 0
                for i in range(len(x_vals)):
                    pred = exp_decay_model(x_vals[i], a, b, c_init)
                    error += (y_vals[i] - pred) ** 2
                
                if error < best_error:
                    best_error = error
                    best_params = (a, b, c_init)
                    
            except (ValueError, OverflowError, ZeroDivisionError):
                continue
        
        # If no valid fit found, use simple exponential without offset
        if best_params is None:
            # Fit y = a * exp(-b * x) (c = 0)
            y_ln = [math.log(max(y, 1e-10)) for y in y_vals]
            n = len(x_vals)
            sum_x = sum(x_vals)
            sum_y_ln = sum(y_ln)
            sum_xy = sum(x_vals[i] * y_ln[i] for i in range(n))
            sum_x2 = sum(x * x for x in x_vals)
            
            b = -(n * sum_xy - sum_x * sum_y_ln) / (n * sum_x2 - sum_x * sum_x)
            ln_a = (sum_y_ln + b * sum_x) / n
            a = math.exp(ln_a)
            best_params = (a, abs(b), 0)
        
        return best_params

    # Fit the model
    a, b, c = fit_exp_decay(x_data, y_data)

    # Make predictions on x = [0, 2, 4, 6, 8, 10]
    eval_x = [0, 2, 4, 6, 8, 10]
    predictions = [exp_decay_model(x, a, b, c) for x in eval_x]

    # Calculate statistics
    mean_pred = sum(predictions) / len(predictions)
    sorted_preds = sorted(predictions)
    n = len(sorted_preds)
    median_pred = (sorted_preds[n//2 - 1] + sorted_preds[n//2]) / 2 if n % 2 == 0 else sorted_preds[n//2]

    variance = sum((p - mean_pred) ** 2 for p in predictions) / len(predictions)
    std_pred = math.sqrt(variance)

    return {
        'fitted_params': {'a': a, 'b': b, 'c': c},
        'predictions': predictions,
        'stats': {'mean': mean_pred, 'median': median_pred, 'std': std_pred}
    }