def evaluate_arcfit_model(spec_string):
    """
    Evaluate an ARCFIT fitted model on query points from a specification string.
    
    Utility: Parses ARCFIT specification format and evaluates fitted models at query points.
             Supports power_law model type with formula: y = a * x^b + c
    
    Args:
        spec_string (str): ARCFIT specification in format 
                          "FITTED:model_type;PARAMS:param=val,...;QUERY:x1,x2,..."
    
    Returns:
        str: JSON string containing list of predicted y values rounded to 6 decimal places
    """
    import json
    import math
    
    # Parse the specification string
    parts = spec_string.split(';')
    
    # Extract model type
    fitted_part = parts[0].split(':')
    model_type = fitted_part[1]
    
    # Extract parameters
    params_part = parts[1].split(':')[1]
    param_pairs = params_part.split(',')
    params = {}
    for pair in param_pairs:
        key, value = pair.split('=')
        params[key] = float(value)
    
    # Extract query points
    query_part = parts[2].split(':')[1]
    query_points = [float(x) for x in query_part.split(',')]
    
    # Evaluate model based on type
    results = []
    
    if model_type == 'power_law':
        # Power law formula: y = a * x^b + c
        a = params['a']
        b = params['b']
        c = params['c']
        
        for x in query_points:
            y = a * (x ** b) + c
            results.append(round(y, 6))
    
    return json.dumps(results)