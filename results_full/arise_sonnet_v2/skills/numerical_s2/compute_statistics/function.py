def compute_statistics(data: list[float]) -> dict[str, float]:
    """
    Compute basic statistical measures for a numerical dataset.
    
    Parameters:
    data: List of numerical values to analyze
    
    Returns:
    Dictionary containing statistical measures:
    - mean: Arithmetic mean
    - median: Middle value when sorted
    - standard deviation: Population standard deviation
    - min: Minimum value
    - max: Maximum value
    - count: Number of data points
    
    If error occurs, returns dict with 'error' key containing error message.
    """
    import math
    
    try:
        if not data:
            return {'error': 'Empty dataset provided'}
        
        if not all(isinstance(x, (int, float)) for x in data):
            return {'error': 'All data points must be numeric'}
        
        # Convert to floats to handle mixed int/float input
        numeric_data = [float(x) for x in data]
        n = len(numeric_data)
        
        # Mean
        mean = sum(numeric_data) / n
        
        # Median
        sorted_data = sorted(numeric_data)
        if n % 2 == 0:
            median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median = sorted_data[n//2]
        
        # Population standard deviation
        variance = sum((x - mean) ** 2 for x in numeric_data) / n
        std_dev = math.sqrt(variance)
        
        # Min and max
        min_val = min(numeric_data)
        max_val = max(numeric_data)
        
        return {
            'mean': mean,
            'median': median,
            'standard deviation': std_dev,
            'min': min_val,
            'max': max_val,
            'count': float(n)
        }
        
    except Exception as e:
        return {'error': f'Error computing statistics: {str(e)}'}
