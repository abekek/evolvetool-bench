def decode_abr_and_hash_names(base64_data):
    """
    Decode ABR (Apache Binary Records) data and compute SHA-256 hash of each record's 'name' field.
    
    Utility: Decodes base64-encoded ABR binary data, extracts records with name/role/id fields,
             computes SHA-256 hash for each name, and returns JSON array of name and hash pairs.
    
    Args:
        base64_data (str): Base64-encoded ABR binary data containing records
    
    Returns:
        str: JSON string containing array of objects with 'name' and 'name_hash' fields
    """
    import base64
    import hashlib
    import json
    
    # Decode base64 data
    binary_data = base64.b64decode(base64_data)
    
    records = []
    offset = 0
    
    while offset < len(binary_data):
        # Check for record marker (0x03 at start, 0xFF at end)
        if binary_data[offset] != 0x03:
            break
            
        offset += 1
        record = {}
        
        # Parse fields until we hit the end marker (0xFF)
        while offset < len(binary_data) and binary_data[offset] != 0xFF:
            # Read field name length
            field_name_len = binary_data[offset]
            offset += 1
            
            # Read field name
            field_name = binary_data[offset:offset + field_name_len].decode('utf-8')
            offset += field_name_len
            
            # Skip field type marker (0x00)
            offset += 1
            
            # Read field value length
            field_value_len = binary_data[offset]
            offset += 1
            
            # Read field value
            field_value = binary_data[offset:offset + field_value_len].decode('utf-8')
            offset += field_value_len
            
            record[field_name] = field_value
        
        # Skip end marker (0xFF)
        if offset < len(binary_data) and binary_data[offset] == 0xFF:
            offset += 1
        
        # Skip record separator (0x03)
        if offset < len(binary_data) and binary_data[offset] == 0x03:
            offset += 1
            
        records.append(record)
    
    # Create result with name and hash
    result = []
    for record in records:
        if 'name' in record:
            name = record['name']
            name_hash = hashlib.sha256(name.encode('utf-8')).hexdigest()
            result.append({
                'name': name,
                'name_hash': name_hash
            })
    
    return json.dumps(result, indent=2)