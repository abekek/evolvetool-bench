def decode_abr_to_json(base64_data):
    """
    Decode ABR (Array-Based Record) format data from base64 and return as JSON array.
    
    ABR format appears to use length-prefixed strings where each field starts with
    a byte indicating the field name length, followed by the field name, then value length,
    then the value. Records are separated by 0xFF bytes.
    
    Args:
        base64_data (str): Base64 encoded ABR format data
        
    Returns:
        str: JSON array string containing decoded records as objects
    """
    import base64
    import json
    
    # Decode base64 data
    try:
        raw_data = base64.b64decode(base64_data)
    except Exception:
        return "[]"
    
    records = []
    i = 0
    
    while i < len(raw_data):
        # Skip separator bytes (0xFF) between records
        if raw_data[i] == 0xFF:
            i += 1
            continue
            
        # Start parsing a new record
        record = {}
        
        # Parse fields until we hit a separator or end of data
        while i < len(raw_data) and raw_data[i] != 0xFF:
            # Skip any control bytes at start of record
            if raw_data[i] in [0x02, 0x03, 0x04]:
                i += 1
                continue
                
            # Read field name length
            if i >= len(raw_data):
                break
            field_name_len = raw_data[i]
            i += 1
            
            # Read field name
            if i + field_name_len > len(raw_data):
                break
            field_name = raw_data[i:i + field_name_len].decode('utf-8', errors='ignore')
            i += field_name_len
            
            # Read value length
            if i >= len(raw_data):
                break
            value_len = raw_data[i]
            i += 1
            
            # Read value
            if i + value_len > len(raw_data):
                break
            value = raw_data[i:i + value_len].decode('utf-8', errors='ignore')
            i += value_len
            
            record[field_name] = value
        
        if record:
            records.append(record)
    
    return json.dumps(records, indent=2)