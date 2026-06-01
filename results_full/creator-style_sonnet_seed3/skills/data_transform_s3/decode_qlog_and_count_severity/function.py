def decode_qlog_and_count_severity(base64_data):
    """
    Decode QLOG binary data from base64 and count log entries by severity level.
    
    QLOG format appears to use severity indicators in the binary data preceding text messages.
    Analyzes message content patterns to determine severity levels.
    
    Args:
        base64_data (str): Base64 encoded QLOG binary data
        
    Returns:
        dict: Mapping of severity level names to their counts
              e.g. {'INFO': 2, 'WARN': 2, 'ERROR': 2}
    """
    import base64
    import re
    
    # Decode base64 data
    binary_data = base64.b64decode(base64_data)
    
    # Extract text messages from binary data
    # Look for printable ASCII sequences that appear to be log messages
    text_pattern = rb'[A-Za-z][A-Za-z0-9\s:.,;!?\-_/]*'
    matches = re.findall(text_pattern, binary_data)
    
    # Filter for actual log messages (longer than 10 chars)
    messages = []
    for match in matches:
        try:
            decoded = match.decode('ascii').strip()
            if len(decoded) > 10:
                messages.append(decoded)
        except UnicodeDecodeError:
            continue
    
    # Count by severity based on message content patterns
    severity_counts = {'INFO': 0, 'WARN': 0, 'ERROR': 0}
    
    for message in messages:
        message_lower = message.lower()
        
        # ERROR indicators
        if any(keyword in message_lower for keyword in ['error', 'failed', 'timeout', 'exception']):
            severity_counts['ERROR'] += 1
        # WARN indicators  
        elif any(keyword in message_lower for keyword in ['warn', 'slow', 'retry', 'retrying']):
            severity_counts['WARN'] += 1
        # INFO indicators (default for neutral messages)
        elif any(keyword in message_lower for keyword in ['start', 'connect', 'establish', 'init']):
            severity_counts['INFO'] += 1
    
    return severity_counts