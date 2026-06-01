def decode_abr_binary_record(base64_data: str) -> list[dict]:
    """Decode ABR (ARISE Binary Record) format data and return as JSON array of objects."""
    import base64
    import traceback
    import sys
    
    try:
        # Step 1: Decode the base64 string to get the raw binary data
        binary_data = base64.b64decode(base64_data)
        
        # Step 2: Parse the binary data to identify record boundaries and field structures
        records = []
        pos = 0
        
        while pos < len(binary_data):
            # Check for record separator (0xFF) or end of data
            if binary_data[pos] == 0xFF:
                pos += 1
                continue
            
            record = {}
            
            # Step 3: Extract field names and values from each record using the ABR format specification
            while pos < len(binary_data) and binary_data[pos] != 0xFF:
                # Read field name length
                if pos >= len(binary_data):
                    break
                name_length = binary_data[pos]
                pos += 1
                
                # Read field name
                if pos + name_length > len(binary_data):
                    break
                field_name = binary_data[pos:pos + name_length].decode('utf-8')
                pos += name_length
                
                # Read null terminator for field name
                if pos < len(binary_data) and binary_data[pos] == 0x00:
                    pos += 1
                
                # Read value length
                if pos >= len(binary_data):
                    break
                value_length = binary_data[pos]
                pos += 1
                
                # Read field value
                if pos + value_length > len(binary_data):
                    break
                field_value = binary_data[pos:pos + value_length].decode('utf-8')
                pos += value_length
                
                # Step 4: Convert the parsed records into a list of dictionary objects with proper data types
                # Try to convert numeric values to appropriate types
                if field_value.isdigit():
                    field_value = int(field_value)
                elif field_value.replace('.', '').isdigit() and field_value.count('.') == 1:
                    field_value = float(field_value)
                
                record[field_name] = field_value
            
            if record:  # Only add non-empty records
                records.append(record)
        
        # Step 5: Return the structured data as a JSON-compatible list of objects
        return records
        
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        return []