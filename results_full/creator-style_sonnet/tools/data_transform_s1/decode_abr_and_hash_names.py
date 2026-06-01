def decode_abr_and_hash_names(base64_data):
    """
    Decode ABR (Apache Binary Record) format data and compute SHA-256 hashes of name fields.

    ABR format analysis from the given data:
    - Type bytes indicate field types (2, 3, 4)
    - Fields are encoded as: type + field_name_length + field_name + null_terminator + value_length + value
    - Records are separated by type 3 (0x03)
    - End of data marked by 0xFF

    Utility: Decodes base64-encoded ABR binary data, extracts records with name/role/id fields,
             computes SHA-256 hash of each record's name field, and returns structured JSON.

    Args:
        base64_data (str): Base64-encoded ABR binary data containing records

    Returns:
        str: JSON array string with objects containing 'name' and 'name_hash' fields
    """
    import base64
    import hashlib
    import json

    # Decode base64 data
    binary_data = base64.b64decode(base64_data)
    
    records = []
    pos = 0

    while pos < len(binary_data):
        # Check for end marker
        if binary_data[pos] == 0xFF:
            pos += 1
            continue
            
        # Check for record separator at start of new record
        if binary_data[pos] == 0x03:
            pos += 1
            
        record = {}
        
        # Parse fields in current record
        while pos < len(binary_data):
            current_byte = binary_data[pos]
            
            # Check for record separator or end marker
            if current_byte == 0x03:
                # This marks end of current record, don't advance pos
                break
            elif current_byte == 0xFF:
                # End of all data
                break
            
            # This should be a field type
            field_type = current_byte
            pos += 1
            
            if field_type in [2, 4]:  # Field types we handle
                # Read field name length
                if pos >= len(binary_data):
                    break
                name_len = binary_data[pos]
                pos += 1
                
                # Read field name
                if pos + name_len > len(binary_data):
                    break
                field_name = binary_data[pos:pos + name_len].decode('utf-8')
                pos += name_len
                
                # Skip null terminator
                if pos < len(binary_data) and binary_data[pos] == 0:
                    pos += 1
                
                # Read value length
                if pos >= len(binary_data):
                    break
                value_len = binary_data[pos]
                pos += 1
                
                # Read field value
                if pos + value_len > len(binary_data):
                    break
                field_value = binary_data[pos:pos + value_len].decode('utf-8')
                pos += value_len
                
                record[field_name] = field_value
            else:
                # Unknown field type, try to skip
                pos += 1
        
        # If we have a complete record with a name field, add it to results
        if record and 'name' in record:
            name = record['name']
            name_hash = hashlib.sha256(name.encode('utf-8')).hexdigest()
            records.append({
                'name': name,
                'name_hash': name_hash
            })
            
        # Move past current position if we're not at a separator or end
        if pos < len(binary_data) and binary_data[pos] not in [0x03, 0xFF]:
            pos += 1

    return json.dumps(records, indent=2)