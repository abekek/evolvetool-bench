def decode_qlog_data(qlog_data: str) -> list[dict]:
    """Decode QLOG formatted data and return structured log records."""
    import base64
    import struct
    import traceback
    import sys
    
    try:
        # Step 1: Base64 decode the input string to get raw binary data
        binary_data = base64.b64decode(qlog_data)
        
        # Step 2: Parse the binary header to extract timestamp, severity, and subsystem information
        # Assuming QLOG format: 4 bytes timestamp, 1 byte severity, 1 byte subsystem, 2 bytes length, then message
        if len(binary_data) < 8:
            raise ValueError("Binary data too short for QLOG header")
        
        # Unpack header: timestamp (4 bytes), severity (1 byte), subsystem (1 byte), length (2 bytes)
        timestamp, severity, subsystem, msg_length = struct.unpack('<IBBH', binary_data[:8])
        
        # Step 3: Extract the message payload from the remaining binary data after the header
        if len(binary_data) < 8 + msg_length:
            raise ValueError("Binary data shorter than expected message length")
        
        message_bytes = binary_data[8:8 + msg_length]
        message = message_bytes.decode('utf-8', errors='replace')
        
        # Step 4: Convert the parsed components into a structured dictionary format
        log_record = {
            'timestamp': timestamp,
            'severity': severity,
            'subsystem': subsystem,
            'message': message,
            'raw_length': msg_length
        }
        
        # Step 5: Return the log record(s) as a list of dictionaries with standardized fields
        return [log_record]
        
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        return []