def parse_qlog(binary_data: bytes) -> list[dict]:
    """
    Parse QLOG binary format data into structured records.
    
    QLOG format per record:
    - timestamp: 8 bytes (uint64, little-endian)
    - flags: 1 byte (uint8)
    - length: 2 bytes (uint16, little-endian) 
    - message: variable length based on length field
    
    Args:
        binary_data: Raw binary data in QLOG format
        
    Returns:
        List of dictionaries with keys: timestamp, flags, length, message, continuation
        On error, returns list with single dict containing 'error' key
    """
    import struct
    import datetime
    
    try:
        # Validate input type
        if not isinstance(binary_data, bytes):
            return [{'error': f'Input must be bytes, got {type(binary_data).__name__}'}]
            
        records = []
        offset = 0
        
        while offset < len(binary_data):
            # Check if we have enough bytes for header (8 + 1 + 2 = 11 bytes)
            if offset + 11 > len(binary_data):
                break
                
            # Parse header: timestamp (8), flags (1), length (2)
            header = struct.unpack('<QBH', binary_data[offset:offset + 11])
            timestamp_raw, flags, msg_length = header
            
            # Check if we have enough bytes for the message
            if offset + 11 + msg_length > len(binary_data):
                records.append({
                    'error': f'Incomplete message at offset {offset}: expected {msg_length} bytes, only {len(binary_data) - offset - 11} available'
                })
                break
                
            # Extract message
            message_start = offset + 11
            message_end = message_start + msg_length
            message_bytes = binary_data[message_start:message_end]
            
            # Try to decode message as UTF-8, fall back to hex if it fails
            try:
                message = message_bytes.decode('utf-8')
            except UnicodeDecodeError:
                message = message_bytes.hex()
                
            # Convert timestamp to readable format (assuming Unix timestamp in microseconds)
            try:
                timestamp_seconds = timestamp_raw / 1000000.0
                timestamp_dt = datetime.datetime.fromtimestamp(timestamp_seconds)
                timestamp_str = timestamp_dt.isoformat()
            except (ValueError, OSError):
                timestamp_str = str(timestamp_raw)
                
            # Parse flags
            continuation = bool(flags & 0x04)  # bit 2 indicates continuation
            
            record = {
                'timestamp': timestamp_raw,  # Keep raw timestamp as primary field
                'timestamp_str': timestamp_str,  # ISO string as separate field
                'flags': flags,
                'length': msg_length,
                'message': message,
                'continuation': continuation
            }
            
            records.append(record)
            offset = message_end
            
        return records
        
    except struct.error as e:
        return [{'error': f'Struct parsing error: {str(e)}'}]
    except Exception as e:
        return [{'error': f'Unexpected error: {str(e)}'}]