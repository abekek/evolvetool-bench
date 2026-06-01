def evaluate_arcfit_model(spec_string):
    """
    Evaluate an ARCFIT fitted model on query points.
    
    Utility: Parses ARCFIT specification string containing model type, parameters, 
             and query points, then evaluates the fitted model and returns predictions.
    
    Args:
        spec_string (str): ARCFIT specification in format 
                          "FITTED:model_type;PARAMS:param=value,...;QUERY:x1,x2,..."
    
    Returns:
        str: JSON string containing list of predicted y values rounded to 6 decimal places
    """
    import json
    import math
    
    # Parse the specification string
    parts = spec_string.split(';')
    
    # Extract model type
    fitted_part = parts[0].split(':')[1]
    model_type = fitted_part.strip()
    
    # Extract parameters
    params_part = parts[1].split(':')[1]
    params = {}
    for param_str in params_part.split(','):
        key, value = param_str.split('=')
        params[key.strip()] = float(value.strip())
    
    # Extract query points
    query_part = parts[2].split(':')[1]
    query_points = [float(x.strip()) for x in query_part.split(',')]
    
    # Evaluate model based on type
    predictions = []
    
    if model_type == 'power_law':
        # Power law: y = a * x^b + c
        a = params['a']
        b = params['b'] 
        c = params['c']
        
        for x in query_points:
            y = a * (x ** b) + c
            predictions.append(round(y, 6))
    
    elif model_type == 'exponential':
        # Exponential: y = a * exp(b * x) + c
        a = params['a']
        b = params['b']
        c = params['c']
        
        for x in query_points:
            y = a * math.exp(b * x) + c
            predictions.append(round(y, 6))
    
    elif model_type == 'logarithmic':
        # Logarithmic: y = a * log(b * x) + c
        a = params['a']
        b = params['b']
        c = params['c']
        
        for x in query_points:
            y = a * math.log(b * x) + c
            predictions.append(round(y, 6))
    
    elif model_type == 'linear':
        # Linear: y = a * x + b
        a = params['a']
        b = params['b']
        
        for x in query_points:
            y = a * x + b
            predictions.append(round(y, 6))
    
    return json.dumps(predictions)