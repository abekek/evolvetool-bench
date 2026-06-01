def fit_arcfit_exp_decay_model(spec_string):
    """
    Fits an exponential decay model (y = a * exp(-b * x) + c) to data and evaluates predictions.
    
    Utility: Parses ARCFIT specification, fits exponential decay model using least squares,
             evaluates on specified x values, and computes statistics of predictions.
    
    Args:
        spec_string (str): ARCFIT specification containing model type, parameters, and data
                          Format: "MODEL:exp_decay;PARAMS:a=?,b=?,c=?;DATA:x1,y1|x2,y2|..."
    
    Returns:
        dict: JSON object with keys:
              - 'fitted_params': dict with fitted a, b, c values
              - 'predictions': list of y values predicted at x = [0, 2, 4, 6, 8, 10]
              - 'stats': dict with mean, median, std of predictions
    """
    import math
    import json
    
    # Parse the specification string
    parts = spec_string.split(';')
    model_part = parts[0]  # MODEL:exp_decay
    params_part = parts[1]  # PARAMS:a=?,b=?,c=?
    data_part = parts[2]   # DATA:...
    
    # Extract data points
    data_str = data_part.split(':', 1)[1]  # Remove "DATA:"
    data_pairs = data_str.split('|')
    
    x_data = []
    y_data = []
    for pair in data_pairs:
        x_val, y_val = pair.split(',')
        x_data.append(float(x_val))
        y_data.append(float(y_val))
    
    n = len(x_data)
    
    # Fit exponential decay model: y = a * exp(-b * x) + c
    # Using linearization: ln(y - c) = ln(a) - b * x
    # We'll use a simple approach assuming c is close to the minimum y value
    
    # Initial estimate for c (asymptotic value)
    c_est = min(y_data) * 0.9
    
    # Try to linearize: ln(y - c) = ln(a) - b * x
    try:
        ln_y_minus_c = [math.log(y - c_est) for y in y_data]
    except ValueError:
        # If log fails, adjust c_est
        c_est = min(y_data) - 0.1
        ln_y_minus_c = [math.log(y - c_est) for y in y_data]
    
    # Linear regression on ln(y - c) vs x
    sum_x = sum(x_data)
    sum_ln_y = sum(ln_y_minus_c)
    sum_x_ln_y = sum(x * ln_y for x, ln_y in zip(x_data, ln_y_minus_c))
    sum_x_sq = sum(x * x for x in x_data)
    
    # Solve for slope (-b) and intercept (ln(a))
    b_est = (n * sum_x_ln_y - sum_x * sum_ln_y) / (sum_x * sum_x - n * sum_x_sq)
    ln_a_est = (sum_ln_y - b_est * sum_x) / n
    a_est = math.exp(ln_a_est)
    b_est = -b_est  # Convert slope to positive decay constant
    
    # Refine c using non-linear approach (simple iteration)
    for _ in range(5):
        # Calculate residuals and adjust c
        residuals = []
        for x, y in zip(x_data, y_data):
            pred = a_est * math.exp(-b_est * x) + c_est
            residuals.append(y - pred)
        
        # Update c based on mean residual
        c_est += sum(residuals) / len(residuals) * 0.1
        
        # Re-fit a and b with new c
        try:
            ln_y_minus_c = [math.log(y - c_est) for y in y_data]
            sum_ln_y = sum(ln_y_minus_c)
            sum_x_ln_y = sum(x * ln_y for x, ln_y in zip(x_data, ln_y_minus_c))
            
            b_est = (n * sum_x_ln_y - sum_x * sum_ln_y) / (sum_x * sum_x - n * sum_x_sq)
            ln_a_est = (sum_ln_y - b_est * sum_x) / n
            a_est = math.exp(ln_a_est)
            b_est = -b_est
        except ValueError:
            break
    
    # Make predictions at x = [0, 2, 4, 6, 8, 10]
    eval_x = [0, 2, 4, 6, 8, 10]
    predictions = []
    for x in eval_x:
        y_pred = a_est * math.exp(-b_est * x) + c_est
        predictions.append(y_pred)
    
    # Compute statistics
    mean_pred = sum(predictions) / len(predictions)
    
    # Median
    sorted_preds = sorted(predictions)
    n_pred = len(sorted_preds)
    if n_pred % 2 == 0:
        median_pred = (sorted_preds[n_pred//2 - 1] + sorted_preds[n_pred//2]) / 2
    else:
        median_pred = sorted_preds[n_pred//2]
    
    # Standard deviation
    variance = sum((p - mean_pred)**2 for p in predictions) / len(predictions)
    std_pred = math.sqrt(variance)