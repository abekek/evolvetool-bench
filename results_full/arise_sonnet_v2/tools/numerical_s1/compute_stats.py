def compute_stats(data: list[float]) -> dict[str, float]:
    """
    Calculate basic statistical measures for a numerical dataset.
    
    Args:
        data: List of numerical values
        
    Returns:
        Dictionary containing statistical measures:
        - mean: Arithmetic mean
        - median: Middle value when sorted
        - standard deviation: Population standard deviation
        - min: Minimum value
        - max: Maximum value
        - count: Number of data points
        Or error information if computation fails
    """
    import math
    
    try:
        if not data:
            return {"error": "Empty dataset provided"}
        
        if not all(isinstance(x, (int, float)) for x in data):
            return {"error": "All data points must be numerical"}
        
        # Check for invalid values (NaN, infinity)
        for x in data:
            if not math.isfinite(float(x)):
                return {"error": "Data contains non-finite values (NaN or infinity)"}
        
        # Convert to float to handle mixed int/float
        data_float = [float(x) for x in data]
        n = len(data_float)
        
        # Mean - use more numerically stable calculation for extreme values
        mean = sum(data_float) / n
        
        # Median
        sorted_data = sorted(data_float)
        if n % 2 == 0:
            median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median = sorted_data[n//2]
        
        # Population standard deviation - use numerically stable two-pass algorithm
        variance = sum((x - mean) ** 2 for x in data_float) / n
        std_dev = math.sqrt(variance)
        
        # Min and max
        min_val = min(data_float)
        max_val = max(data_float)
        
        return {
            "mean": float(mean),
            "median": float(median),
            "standard deviation": float(std_dev),
            "min": float(min_val),
            "max": float(max_val),
            "count": n
        }
        
    except Exception as e:
        return {"error": f"Computation failed: {str(e)}"}