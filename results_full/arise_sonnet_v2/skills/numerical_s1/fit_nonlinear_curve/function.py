def fit_nonlinear_curve(model_type: str, data_points: list, initial_params: dict) -> dict:
    """
    Perform non-linear curve fitting using least squares optimization.
    
    Args:
        model_type: Type of model to fit ('exponential_decay', 'exponential_growth', 'power', 'gaussian')
        data_points: List of [x, y] coordinate pairs
        initial_params: Dictionary of initial parameter guesses for the model
    
    Returns:
        Dictionary containing:
        - 'success': bool indicating if fitting succeeded
        - 'parameters': dict of fitted parameter values (if successful)
        - 'r_squared': coefficient of determination (if successful)
        - 'residual_sum_squares': sum of squared residuals (if successful)
        - 'error': error message (if failed)
    """
    import math
    from scipy.optimize import curve_fit
    import numpy as np
    
    # Validate model type first and raise exception for invalid types
    model_functions = {
        'exponential_decay': lambda x, a, b, c: a * np.exp(-b * x) + c,
        'exponential_growth': lambda x, a, b, c: a * np.exp(b * x) + c,
        'power': lambda x, a, b, c: np.where(x <= 0, c, a * np.power(x, b) + c),
        'gaussian': lambda x, a, mu, sigma, c: a * np.exp(-0.5 * ((x - mu) / np.maximum(sigma, 1e-10)) ** 2) + c
    }
    
    if model_type not in model_functions:
        raise ValueError(f'Unknown model type: {model_type}')
    
    try:
        # Validate inputs
        if not data_points or len(data_points) < 2:
            return {'success': False, 'error': 'Need at least 2 data points'}
        
        if not all(len(point) == 2 for point in data_points):
            return {'success': False, 'error': 'All data points must be [x, y] pairs'}
        
        # Extract x and y values
        x_vals = np.array([float(point[0]) for point in data_points])
        y_vals = np.array([float(point[1]) for point in data_points])
        
        # Validate initial parameters based on model type
        required_params = {
            'exponential_decay': ['a', 'b', 'c'],
            'exponential_growth': ['a', 'b', 'c'],
            'power': ['a', 'b', 'c'],
            'gaussian': ['a', 'mu', 'sigma', 'c']
        }
        
        if not all(param in initial_params for param in required_params[model_type]):
            return {'success': False, 'error': f'Missing required parameters for {model_type}: {required_params[model_type]}'}
        
        # Get model function and initial parameter values
        model_func = model_functions[model_type]
        param_names = required_params[model_type]
        p0 = [initial_params[param] for param in param_names]
        
        # Perform curve fitting using scipy's curve_fit
        try:
            popt, pcov = curve_fit(model_func, x_vals, y_vals, p0=p0, maxfev=5000)
        except Exception as e:
            # Fall back to simple optimization if scipy fails
            return _fallback_optimization(model_type, x_vals, y_vals, initial_params, required_params)
        
        # Create parameter dictionary
        fitted_params = {param_names[i]: popt[i] for i in range(len(param_names))}
        
        # Compute predictions and residuals
        y_pred = model_func(x_vals, *popt)
        residuals = y_vals - y_pred
        rss = np.sum(residuals ** 2)
        
        # Calculate R-squared
        y_mean = np.mean(y_vals)
        tss = np.sum((y_vals - y_mean) ** 2)
        r_squared = 1 - (rss / tss) if tss > 0 else 0
        
        return {
            'success': True,
            'parameters': fitted_params,
            'r_squared': float(r_squared),
            'residual_sum_squares': float(rss)
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}

def _fallback_optimization(model_type, x_vals, y_vals, initial_params, required_params):
    """Fallback optimization using simple gradient descent."""
    import math
    import numpy as np
    
    # Define model functions for fallback
    def exponential_decay(x, params):
        a, b, c = params['a'], params['b'], params['c']
        return a * math.exp(-b * x) + c
    
    def exponential_growth(x, params):
        a, b, c = params['a'], params['b'], params['c']
        return a * math.exp(b * x) + c
    
    def power_law(x, params):
        a, b, c = params['a'], params['b'], params['c']
        if x <= 0:
            return c
        return a * (x ** b) + c
    
    def gaussian(x, params):
        a, mu, sigma, c = params['a'], params['mu'], params['sigma'], params['c']
        if abs(sigma) < 1e-10:
            return c
        return a * math.exp(-0.5 * ((x - mu) / sigma) ** 2) + c
    
    # Select model function
    fallback_functions = {
        'exponential_decay': exponential_decay,
        'exponential_growth': exponential_growth,
        'power': power_law,
        'gaussian': gaussian
    }
    
    model_func = fallback_functions[model_type]
    
    # Simple gradient descent optimization
    params = initial_params.copy()
    learning_rate = 0.001
    max_iterations = 1000
    tolerance = 1e-8
    
    def compute_residuals(params):
        residuals = []
        for x, y in zip(x_vals, y_vals):
            try:
                predicted = model_func(x, params)
                if math.isnan(predicted) or math.isinf(predicted):
                    return None
                residuals.append(y - predicted)
            except (ValueError, OverflowError, ZeroDivisionError):
                return None
        return residuals
    
    def compute_cost(params):
        residuals = compute_residuals(params)
        if residuals is None:
            return float('inf')
        return sum(r * r for r in residuals)
    
    # Numerical gradient computation
    def compute_gradient(params, h=1e-6):
        gradient = {}
        base_cost = compute_cost(params)
        if math.isinf(base_cost):
            return None
        
        for param_name in params:
            params_plus = params.copy()
            params_plus[param_name] += h
            cost_plus = compute_cost(params_plus)
            
            if math.isinf(cost_plus):
                gradient[param_name] = 0
            else:
                gradient[param_name] = (cost_plus - base_cost) / h
        
        return gradient
    
    # Optimization loop
    prev_cost = float('inf')
    
    for iteration in range(max_iterations):
        current_cost = compute_cost(params)
        
        if math.isinf(current_cost):
            return {'success': False, 'error': 'Model evaluation failed during optimization'}
        
        # Check for convergence
        if abs(prev_cost - current_cost) < tolerance:
            break
        
        # Compute gradient
        gradient = compute_gradient(params)
        if gradient is None:
            return {'success': False, 'error': 'Gradient computation failed'}
        
        # Update parameters
        for param_name in params:
            params[param_name] -= learning_rate * gradient[param_name]
        
        prev_cost = current_cost
    
    # Compute final statistics
    final_residuals = compute_residuals(params)
    if final_residuals is None:
        return {'success': False, 'error': 'Final model evaluation failed'}
    
    rss = sum(r * r for r in final_residuals)
    
    # Calculate R-squared
    y_mean = sum(y_vals) / len(y_vals)
    tss = sum((y - y_mean) ** 2 for y in y_vals)
    r_squared = 1 - (rss / tss) if tss > 0 else 0
    
    return {
        'success': True,
        'parameters': params,
        'r_squared': r_squared,
        'residual_sum_squares': rss
    }