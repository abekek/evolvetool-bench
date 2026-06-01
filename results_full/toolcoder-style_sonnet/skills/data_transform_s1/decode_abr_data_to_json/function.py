def decode_abr_data_to_json(abr_data: str) -> list[dict]:
    """Decode base64-encoded ABR (Application Binary Record) data and return as JSON array."""
    import base64
    import sys
    import traceback
    
    try:
        # Step 1: Decode the base64 string to get raw binary data
        binary_data = base64.b64decode(abr_data)
        
        # Step 2: Parse the binary data to identify record boundaries and field separators
        records = []
        pos = 0
        
        while pos < len(binary_data):
            # Check for record start marker (0x02)
            if binary_data[pos] == 0x02:
                pos += 1  # Skip record start marker
                record = {}
                
                # Step 3: Extract individual records and their field-value pairs from the binary stream
                while pos < len(binary_data):
                    # Check for record end marker (0xFF) or next record start (0x02)
                    if binary_data[pos] == 0xFF:
                        pos += 1  # Skip record end marker
                        break
                    elif binary_data[pos] == 0x02:
                        # Start of next record, don't consume this byte
                        break
                    
                    # Read field name length
                    field_name_length = binary_data[pos]
                    pos += 1
                    
                    # Read field name
                    field_name = binary_data[pos:pos + field_name_length].decode('utf-8')
                    pos += field_name_length
                    
                    # Read field value length
                    field_value_length = binary_data[pos]
                    pos += 1
                    
                    # Read field value
                    field_value = binary_data[pos:pos + field_value_length].decode('utf-8')
                    pos += field_value_length
                    
                    # Step 4: Convert each record's fields into dictionary format, handling data type conversion
                    # Try to convert numeric values
                    try:
                        # Try integer first
                        if '.' not in field_value:
                            converted_value = int(field_value)
                        else:
                            # Try float
                            converted_value = float(field_value)
                    except ValueError:
                        # Keep as string if not numeric
                        converted_value = field_value
                    
                    record[field_name] = converted_value
                
                if record:  # Only add non-empty records
                    records.append(record)
            else:
                pos += 1  # Skip unknown bytes
        
        # Step 5: Collect all parsed records into a list and return as JSON-serializable structure
        return records
        
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        raise