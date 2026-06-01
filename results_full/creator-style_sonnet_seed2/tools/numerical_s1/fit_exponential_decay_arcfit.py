def fit_exponential_decay_arcfit(data_string, tolerance=0.5):
    """
    Fits an exponential decay model to noisy data with robust handling of near-degenerate cases.
    
    Utility: Parses ARCFIT spec data and fits exp_decay model y = a*exp(-b*x) + c, 
             handling cases where exponential signal is very small relative to offset.
             Uses multiple initialization strategies and bounds to prevent crashes.
    
    Args:
        data_string (str): Pipe-separated data points in format "x1,y1|x2,y2|..."
        tolerance (float): Acceptable tolerance for offset parameter validation
    
    Returns:
        str: JSON string with fitted parameters a, b, c to 6 decimal places
    """
    import json
    import math
    
    # Parse data
    points = data_string.split('|')
    x_data = []
    y_data = []
    
    for point in points:
        x_str, y_str = point.split(',')
        x_data.append(float(x_str))
        y_data.append(float(y_str))
    
    # Simple statistics for initialization
    y_min = min(y_data)
    y_max = max(y_data)
    y_mean = sum(y_data) / len(y_data)
    
    # Multiple initialization strategies for robustness
    init_strategies = [
        # Strategy 1: Small exponential component
        {'a': (y_max - y_min), 'b': 0.1, 'c': y_min},
        # Strategy 2: Mean-based
        {'a': y_mean * 0.01, 'b': 0.01, 'c': y_mean},
        # Strategy 3: Conservative
        {'a': 1.0, 'b': 0.001, 'c': y_mean},
        # Strategy 4: Assume mostly constant
        {'a': 0.1, 'b': 1.0, 'c': y_mean}
    ]
    
    def exp_decay(x, a, b, c):
        try:
            if abs(b * x) > 100:  # Prevent overflow
                return c
            return a * math.exp(-b * x) + c
        except (OverflowError, ValueError):
            return c
    
    def residual_sum_squares(params, x_data, y_data):
        a, b, c = params
        if b < 0:  # Ensure decay
            return float('inf')
        
        rss = 0
        for i in range(len(x_data)):
            pred = exp_decay(x_data[i], a, b, c)
            rss += (y_data[i] - pred) ** 2
        return rss
    
    # Simple gradient descent with bounds
    def fit_with_bounds(init_params, learning_rate=0.001, max_iter=1000):
        params = [init_params['a'], init_params['b'], init_params['c']]
        best_params = params[:]
        best_rss = residual_sum_squares(params, x_data, y_data)
        
        for iteration in range(max_iter):
            current_rss = residual_sum_squares(params, x_data, y_data)
            
            if current_rss < best_rss:
                best_rss = current_rss
                best_params = params[:]
            
            # Numerical gradient
            epsilon = 1e-8
            gradients = []
            
            for i in range(3):
                params_plus = params[:]
                params_plus[i] += epsilon
                rss_plus = residual_sum_squares(params_plus, x_data, y_data)
                
                params_minus = params[:]
                params_minus[i] -= epsilon
                rss_minus = residual_sum_squares(params_minus, x_data, y_data)
                
                grad = (rss_plus - rss_minus) / (2 * epsilon)
                gradients.append(grad)
            
            # Update with bounds
            params[0] = max(-1000, min(1000, params[0] - learning_rate * gradients[0]))  # a bounds
            params[1] = max(1e-10, min(100, params[1] - learning_rate * gradients[1]))   # b bounds (positive)
            params[2] = max(0, min(200, params[2] - learning_rate * gradients[2]))       # c bounds
            
            # Early stopping
            if iteration > 10 and abs(current_rss - best_rss) < 1e-12:
                break
        
        return best_params, best_rss
    
    # Try all initialization strategies
    best_overall_params = None
    best_overall_rss = float('inf')
    
    for init in init_strategies:
        try:
            params, rss = fit_with_bounds(init)
            if rss < best_overall_rss:
                best_overall_rss = rss
                best_overall_params = params
        except:
            continue
    
    # Fallback if all strategies fail
    if best_overall_params is None:
        best_overall_params = [0.0, 0.001, y_mean]
    
    # Format result
    result = {
        "a": round(best_overall_params[0], 6),
        "b": round(best_overall_params[1], 6),
        "c": round(best_overall_params[2], 6)
    }
    
    return json.dumps(result)