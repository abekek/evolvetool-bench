def decode_abr_data(base64_data):
    """
    Decode ABR (Attribute-Based Record) format data from base64 string.
    
    ABR format handles records with field-value pairs, properly managing
    delimiter-like bytes (0xFF, 0xFE) and null bytes within field values.
    
    Utility: Decodes ABR binary format that may contain embedded delimiter bytes
    Args: base64_data (str) - Base64 encoded ABR data string
    Returns: list - JSON-serializable list of dictionaries representing records
    """
    import base64
    import json
    
    # Decode base64 to get raw bytes
    raw_data = base64.b64decode(base64_data)
    
    records = []
    pos = 0
    
    while pos < len(raw_data):
        # Check for record separator (0xFF at start of record boundary)
        if pos > 0 and raw_data[pos] == 0xFF:
            pos += 1
            if pos >= len(raw_data):
                break
        
        record = {}
        
        # Parse fields in current record until we hit record boundary or end
        while pos < len(raw_data):
            # Look for field type indicator
            if raw_data[pos] == 0xFF:
                # This marks end of current record
                break
                
            field_type = raw_data[pos]
            pos += 1
            
            if pos >= len(raw_data):
                break
                
            # Read field name (null-terminated)
            field_name = b''
            while pos < len(raw_data) and raw_data[pos] != 0x00:
                field_name += bytes([raw_data[pos]])
                pos += 1
            pos += 1  # Skip null terminator
            
            if pos >= len(raw_data):
                break
                
            # Read field length (2 bytes, big-endian)
            if pos + 1 >= len(raw_data):
                break
            field_length = (raw_data[pos] << 8) | raw_data[pos + 1]
            pos += 2
            
            # Read field value
            if pos + field_length > len(raw_data):
                field_length = len(raw_data) - pos
                
            field_value = raw_data[pos:pos + field_length]
            pos += field_length
            
            # Convert to string, preserving special bytes
            try:
                # Try UTF-8 first
                value_str = field_value.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1 which preserves all byte values
                value_str = field_value.decode('latin-1')
            
            record[field_name.decode('utf-8', errors='replace')] = value_str
        
        if record:  # Only add non-empty records
            records.append(record)
    
    return records