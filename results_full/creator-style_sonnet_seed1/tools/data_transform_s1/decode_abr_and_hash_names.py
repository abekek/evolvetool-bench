def decode_abr_and_hash_names(base64_data):
    """
    Decode ABR (Apache Avro Binary Record) data and compute SHA-256 hashes of name fields.
    
    Utility: Decodes base64-encoded ABR data, extracts records with name/role/id fields,
             computes SHA-256 hash of each name field, and returns structured JSON data.
    
    Args:
        base64_data (str): Base64-encoded ABR data containing records with name, role, and id fields
    
    Returns:
        str: JSON string containing array of objects with 'name' and 'name_hash' fields
    """
    import base64
    import hashlib
    import json
    
    # Decode base64 data
    binary_data = base64.b64decode(base64_data)
    
    # Parse ABR format manually
    records = []
    pos = 0
    
    while pos < len(binary_data):
        if pos + 1 >= len(binary_data):
            break
            
        # Skip record separator if present
        if binary_data[pos] == 0xFF:
            pos += 1
            continue
            
        record = {}
        
        # Parse fields in the record
        while pos < len(binary_data) and binary_data[pos] != 0xFF:
            if pos >= len(binary_data):
                break
                
            # Read field type (assuming 0x03 = start of record, 0x04 = string field, 0x02 = other field)
            field_type = binary_data[pos]
            pos += 1
            
            if field_type == 0x03:  # Record start marker
                continue
            elif field_type == 0x04:  # String field marker
                # Read field name length and name
                if pos >= len(binary_data):
                    break
                name_len = binary_data[pos]
                pos += 1
                
                if pos + name_len > len(binary_data):
                    break
                field_name = binary_data[pos:pos + name_len].decode('utf-8')
                pos += name_len
                
                # Skip null terminator
                if pos < len(binary_data) and binary_data[pos] == 0x00:
                    pos += 1
                
                # Read value length and value
                if pos >= len(binary_data):
                    break
                value_len = binary_data[pos]
                pos += 1
                
                if pos + value_len > len(binary_data):
                    break
                field_value = binary_data[pos:pos + value_len].decode('utf-8')
                pos += value_len
                
                record[field_name] = field_value
                
            elif field_type == 0x02:  # Other field type (like id)
                # Read field name length and name
                if pos >= len(binary_data):
                    break
                name_len = binary_data[pos]
                pos += 1
                
                if pos + name_len > len(binary_data):
                    break
                field_name = binary_data[pos:pos + name_len].decode('utf-8')
                pos += name_len
                
                # Skip null terminator
                if pos < len(binary_data) and binary_data[pos] == 0x00:
                    pos += 1
                
                # Read value length and value
                if pos >= len(binary_data):
                    break
                value_len = binary_data[pos]
                pos += 1
                
                if pos + value_len > len(binary_data):
                    break
                field_value = binary_data[pos:pos + value_len].decode('utf-8')
                pos += value_len
                
                record[field_name] = field_value
            else:
                pos += 1
        
        # Add completed record if it has a name field
        if record and 'name' in record:
            records.append(record)
    
    # Create result with name and name_hash
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