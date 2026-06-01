def evaluate_arcfit_exponential_decay(spec_string):
    """
    Evaluates an ARCFIT exponential decay model at specified x values with robust error handling.
    
    The function parses a specification string containing model type, parameters, and query points,
    then evaluates the exponential decay function y = a * exp(-b * x) + c at each query point.
    Handles edge cases like negative inputs, zero, and large values that cause underflow.
    
    Args:
        spec_string (str): Specification in format "FITTED:model_type;PARAMS:param_assignments;QUERY:x_values"
                          Example: "FITTED:exp_decay;PARAMS:a=1.0,b=1.0,c=0.0;QUERY:0.0,-1.0,100.0"
    
    Returns:
        str: JSON string containing list of predicted y values rounded to 6 decimal places
    """
    import json
    import math
    
    # Parse the specification string
    parts = spec_string.split(';')
    
    # Extract model type
    model_type = parts[0].split(':')[1]
    
    # Extract parameters
    params_str = parts[1].split(':')[1]
    params = {}
    for param_pair in params_str.split(','):
        key, value = param_pair.split('=')
        params[key] = float(value)
    
    # Extract query points
    query_str = parts[2].split(':')[1]
    x_values = [float(x) for x in query_str.split(',')]
    
    # Evaluate exponential decay model: y = a * exp(-b * x) + c
    results = []
    
    for x in x_values:
        try:
            # Calculate -b * x
            exponent = -params['b'] * x
            
            # Handle extreme cases to prevent overflow/underflow
            if exponent > 700:  # exp would overflow
                exp_term = float('inf')
            elif exponent < -700:  # exp would underflow to 0
                exp_term = 0.0
            else:
                exp_term = math.exp(exponent)
            
            # Calculate final result
            y = params['a'] * exp_term + params['c']
            
            # Handle potential infinity or NaN
            if math.isnan(y) or math.isinf(y):
                y = 0.0
            
            # Round to 6 decimal places
            y = round(y, 6)
            
        except (OverflowError, ValueError, ZeroDivisionError):
            # Fallback to 0 for any mathematical errors
            y = 0.0
        
        results.append(y)
    
    return json.dumps(results)