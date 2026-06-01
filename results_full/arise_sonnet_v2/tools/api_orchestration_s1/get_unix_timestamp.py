def get_unix_timestamp() -> int:
    """Get the current Unix timestamp as an integer.
    
    Returns the number of seconds since January 1, 1970 UTC (Unix epoch)
    as an integer. This is commonly used for authentication tokens,
    time-sensitive operations, and API requests that require timestamps.
    
    Returns:
        int: Current Unix timestamp in seconds since epoch
        
    Example:
        >>> timestamp = get_unix_timestamp()
        >>> isinstance(timestamp, int)
        True
        >>> timestamp > 0
        True
    """
    import time
    
    # Get current time as float and convert to integer
    # time.time() is very reliable and shouldn't raise exceptions
    current_time = time.time()
    return int(current_time)