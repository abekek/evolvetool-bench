def fit_arcfit_power_law_model(spec_string):
    """
    Parse an ARCFIT model specification and fit a power law model to the provided data.
    
    Utility: Parses ARCFIT specification format, extracts data points, and fits a power law 
    model (y = a * x^b + c) using least squares optimization. Returns fitted parameters 
    rounded to 6 decimal places.
    
    Args:
        spec_string (str): ARCFIT specification in format 
                          "MODEL:power_law;PARAMS:a=?,b=?,c=?;DATA:x1,y1|x2,y2|..."
    
    Returns:
        dict: JSON object containing fitted parameter values {'a': float, 'b': float, 'c': float}
    """
    import math
    import json
    
    # Parse the specification string
    parts = spec_string.split(';')
    data_part = None
    
    for part in parts:
        if part.startswith('DATA:'):
            data_part = part[5:]  # Remove 'DATA:' prefix
            break
    
    if not data_part:
        return {"error": "No data found in specification"}
    
    # Extract data points
    points = []
    for point_str in data_part.split('|'):
        x_str, y_str = point_str.split(',')
        points.append((float(x_str), float(y_str)))
    
    if len(points) < 3:
        return {"error": "Need at least 3 data points to fit 3 parameters"}
    
    # Simple iterative fitting for power law: y = a * x^b + c
    # Using a basic grid search and least squares minimization
    best_params = None
    min_error = float('inf')
    
    # Grid search over reasonable parameter ranges
    a_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    b_range = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    c_range = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    for a in a_range:
        for b in b_range:
            for c in c_range:
                error = 0.0
                try:
                    for x, y_actual in points:
                        if x <= 0 and b != int(b):
                            error = float('inf')
                            break
                        y_pred = a * (x ** b) + c
                        error += (y_actual - y_pred) ** 2
                    
                    if error < min_error:
                        min_error = error
                        best_params = (a, b, c)
                except:
                    continue
    
    if best_params is None:
        return {"error": "Could not fit model to data"}
    
    # Refine the best parameters with smaller steps around the best solution
    a_best, b_best, c_best = best_params
    
    # Fine-tune around best solution
    for _ in range(10):  # Multiple refinement iterations
        best_local = None
        min_local_error = min_error
        
        step_size = 0.1
        for da in [-step_size, 0, step_size]:
            for db in [-step_size*0.1, 0, step_size*0.1]:
                for dc in [-step_size, 0, step_size]:
                    a_test = a_best + da
                    b_test = b_best + db
                    c_test = c_best + dc
                    
                    if a_test <= 0:
                        continue
                        
                    error = 0.0
                    try:
                        for x, y_actual in points:
                            y_pred = a_test * (x ** b_test) + c_test
                            error += (y_actual - y_pred) ** 2
                        
                        if error < min_local_error:
                            min_local_error = error
                            best_local = (a_test, b_test, c_test)
                    except:
                        continue
        
        if best_local:
            a_best, b_best, c_best = best_local
            min_error = min_local_error
        else:
            break
    
    # Round to 6 decimal places
    result = {
        "a": round(a_best, 6),
        "b": round(b_best, 6), 
        "c": round(c_best, 6)
    }
    
    return result