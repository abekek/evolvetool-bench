def decode_abr_binary_data(base64_data):
    """
    Decode ABR (custom binary) format data from base64 string to JSON array.
    
    Handles edge cases where field values may contain delimiter bytes (0xFF) 
    or null bytes that could be mistaken for field separators.
    
    Args:
        base64_data (str): Base64 encoded ABR binary data
        
    Returns:
        list: JSON array of objects with decoded field-value pairs
    """
    import base64
    import json
    
    # Decode base64 data
    try:
        binary_data = base64.b64decode(base64_data)
    except:
        return []
    
    results = []
    pos = 0
    
    while pos < len(binary_data):
        # Check if we have enough bytes for a record
        if pos >= len(binary_data):
            break
            
        # First byte appears to be record type or count
        if pos + 1 >= len(binary_data):
            break
            
        record_info = binary_data[pos]
        pos += 1
        
        record = {}
        
        # Parse fields within this record
        while pos < len(binary_data):
            # Look for field length indicator
            if pos >= len(binary_data):
                break
                
            field_len = binary_data[pos]
            pos += 1
            
            # If field length is 0xFF, this might be a record separator
            if field_len == 0xFF:
                break
                
            # Extract field name
            if pos + field_len > len(binary_data):
                break
                
            field_name = binary_data[pos:pos + field_len].decode('utf-8', errors='ignore')
            pos += field_len
            
            # Skip null byte separator if present
            if pos < len(binary_data) and binary_data[pos] == 0x00:
                pos += 1
            
            # Get value length
            if pos >= len(binary_data):
                break
                
            value_len = binary_data[pos]
            pos += 1
            
            # Extract value, handling potential embedded delimiters
            if pos + value_len > len(binary_data):
                # Take remaining bytes as value
                value_bytes = binary_data[pos:]
                pos = len(binary_data)
            else:
                value_bytes = binary_data[pos:pos + value_len]
                pos += value_len
            
            # Decode value, handling potential null bytes and special chars
            try:
                # Try UTF-8 first
                value = value_bytes.decode('utf-8', errors='replace')
                # Clean up any null bytes or non-printable chars
                value = ''.join(c for c in value if c.isprintable() or c.isspace())
            except:
                # Fallback to hex representation for binary data
                value = value_bytes.hex()
            
            record[field_name] = value
            
            # Skip trailing null byte if present
            if pos < len(binary_data) and binary_data[pos] == 0x00:
                pos += 1
        
        if record:  # Only add non-empty records
            results.append(record)
    
    return results