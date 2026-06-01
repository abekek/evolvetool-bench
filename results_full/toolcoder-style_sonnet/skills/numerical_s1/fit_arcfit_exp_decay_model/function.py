def fit_arcfit_exp_decay_model(model_spec: str) -> dict:
    """Fit an exponential decay model y = a * exp(-b * x) + c to the given data points and return fitted parameters."""
    import re
    import math
    import traceback
    import sys
    
    try:
        # Step 1: Parse the model specification string to extract the data points (x, y pairs)
        data_match = re.search(r'DATA:([^;]+)', model_spec)
        if not data_match:
            raise ValueError("No DATA section found in model specification")
        
        data_str = data_match.group(1)
        pairs = data_str.split('|')
        data_points = []
        for pair in pairs:
            x_str, y_str = pair.split(',')
            data_points.append((float(x_str), float(y_str)))
        
        x_values = [point[0] for point in data_points]
        y_values = [point[1] for point in data_points]
        n = len(data_points)
        
        # Step 2: Set up the exponential decay fitting problem by transforming to linear form ln(y - c) = ln(a) - b*x
        # Step 3: Use iterative approach to find optimal parameter c, then solve for a and b using linear regression on transformed data
        
        # Find reasonable bounds for c (must be less than min(y) for decay model)
        min_y = min(y_values)
        max_y = max(y_values)
        
        best_params = None
        best_sse = float('inf')
        
        # Try different values of c
        c_candidates = []
        for i in range(100):
            c = min_y - 0.1 - (i / 99.0) * (max_y - min_y + 1.0)
            c_candidates.append(c)
        
        # Step 4: For each trial value of c, ensure y - c > 0 for all points, then perform linear fit on ln(y - c) vs x
        for c in c_candidates:
            # Check if y - c > 0 for all points
            if all(y - c > 0 for y in y_values):
                try:
                    # Transform to linear form: ln(y - c) = ln(a) - b*x
                    ln_y_minus_c = [math.log(y - c) for y in y_values]
                    
                    # Linear regression: ln_y_minus_c = intercept - b*x
                    # Calculate means
                    x_mean = sum(x_values) / n
                    ln_y_mean = sum(ln_y_minus_c) / n
                    
                    # Calculate slope (b) and intercept (ln(a))
                    numerator = sum((x_values[i] - x_mean) * (ln_y_minus_c[i] - ln_y_mean) for i in range(n))
                    denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
                    
                    if denominator == 0:
                        continue
                        
                    b = -numerator / denominator  # negative because we want decay
                    ln_a = ln_y_mean + b * x_mean
                    a = math.exp(ln_a)
                    
                    # Step 5: Select the parameter set (a, b, c) that minimizes the residual sum of squares on the original exponential model
                    # Calculate SSE for original model y = a * exp(-b * x) + c
                    sse = 0
                    for i in range(n):
                        predicted = a * math.exp(-b * x_values[i]) + c
                        sse += (y_values[i] - predicted) ** 2
                    
                    if sse < best_sse:
                        best_sse = sse
                        best_params = (a, b, c)
                        
                except (ValueError, OverflowError):
                    # Skip this c value if it causes numerical issues
                    continue
        
        if best_params is None:
            # Fallback: simple approach
            c = min_y * 0.1
            ln_y_minus_c = [math.log(max(y - c, 1e-10)) for y in y_values]
            
            x_mean = sum(x_values) / n
            ln_y_mean = sum(ln_y_minus_c) / n
            
            numerator = sum((x_values[i] - x_mean) * (ln_y_minus_c[i] - ln_y_mean) for i in range(n))
            denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
            
            if denominator != 0:
                b = -numerator / denominator
                ln_a = ln_y_mean + b * x_mean
                a = math.exp(ln_a)
                best_params = (a, b, c)
            else:
                best_params = (1.0, 0.1, 0.0)
        
        # Step 6: Round the fitted parameters to 6 decimal places and return as JSON object with keys 'a', 'b', 'c'
        a, b, c = best_params
        return {
            'a': round(a, 6),
            'b': round(b, 6),
            'c': round(c, 6)
        }
        
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return {'a': 1.0, 'b': 0.1, 'c': 0.0}