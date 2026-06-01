def decode_qlog_binary_data(base64_data):
    """
    Decode QLOG (Quantized Log Format) binary data into structured log records.
    
    Utility: Parses base64-encoded QLOG binary format into readable log entries with 
    timestamps, severity levels, subsystem IDs, and messages. Handles variable-length 
    payloads and separator markers between entries.
    
    Args:
        base64_data (str): Base64 encoded QLOG binary data
    
    Returns:
        list: List of dictionaries with keys:
            - timestamp (str): ISO format timestamp
            - severity (str): Severity level name (TRACE, DEBUG, INFO, WARN, ERROR, FATAL)  
            - subsystem (int): Subsystem ID (0-15)
            - message (str): UTF-8 decoded message text
    """
    import base64
    import struct
    from datetime import datetime, timezone
    
    # Decode base64 data
    binary_data = base64.b64decode(base64_data)
    
    # QLOG epoch: 2025-01-01 00:00:00 UTC
    qlog_epoch = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    # Severity level mapping
    severity_names = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL']
    
    records = []
    offset = 0
    
    while offset < len(binary_data):
        # Skip separator markers (0xFE 0xFE)
        while offset < len(binary_data) - 1 and binary_data[offset] == 0xFE and binary_data[offset + 1] == 0xFE:
            offset += 2
        
        # Check if we have enough bytes for header
        if offset + 8 > len(binary_data):
            break
            
        # Parse 8-byte header
        header = binary_data[offset:offset + 8]
        
        # Bytes 0-3: uint32 big-endian timestamp
        timestamp_offset = struct.unpack('>I', header[0:4])[0]
        timestamp = qlog_epoch.timestamp() + timestamp_offset
        timestamp_iso = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
        
        # Byte 4: packed severity and subsystem
        packed_byte = header[4]
        severity_level = (packed_byte >> 4) & 0x0F
        subsystem_id = packed_byte & 0x0F
        
        # Byte 5: flags (not used in this implementation)
        flags = header[5]
        
        # Bytes 6-7: uint16 big-endian payload length
        payload_length = struct.unpack('>H', header[6:8])[0]
        
        # Extract payload
        payload_start = offset + 8
        payload_end = payload_start + payload_length
        
        if payload_end > len(binary_data):
            break
            
        payload = binary_data[payload_start:payload_end]
        message = payload.decode('utf-8')
        
        # Create record
        record = {
            'timestamp': timestamp_iso,
            'severity': severity_names[severity_level] if severity_level < len(severity_names) else f'UNKNOWN({severity_level})',
            'subsystem': subsystem_id,
            'message': message
        }
        
        records.append(record)
        offset = payload_end
    
    return records