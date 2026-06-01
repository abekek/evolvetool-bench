def deserialize_and_filter_tpack_data(base64_data):
    """
    Deserialize TPACK format data from base64 string and filter records where 'available' is True.
    
    TPACK appears to be a custom binary format that stores structured data with type markers.
    This function decodes the base64 data, parses the binary TPACK format, and filters results.
    
    Args:
        base64_data (str): Base64 encoded TPACK binary data containing product records
        
    Returns:
        list: List of dictionaries representing records where 'available' field is True
    """
    import base64
    import struct
    
    # Decode base64 data
    binary_data = base64.b64decode(base64_data)
    
    records = []
    offset = 0
    
    while offset < len(binary_data):
        try:
            # Skip initial marker if present
            if offset < len(binary_data) and binary_data[offset] == 0x40:  # '@' marker
                offset += 1
                continue
            
            # Parse record
            record = {}
            
            # Read fields until we hit next record or end
            while offset < len(binary_data):
                if offset >= len(binary_data) - 1:
                    break
                    
                # Check for record separator or start of new record
                if binary_data[offset] == 0x40 and offset > 0:  # '@' at start of new record
                    break
                
                # Read field name length and name
                if offset >= len(binary_data):
                    break
                    
                field_len = binary_data[offset]
                offset += 1
                
                if offset + field_len > len(binary_data):
                    break
                    
                field_name = binary_data[offset:offset + field_len].decode('utf-8', errors='ignore')
                offset += field_len
                
                # Read field value based on type marker
                if offset >= len(binary_data):
                    break
                    
                type_marker = binary_data[offset]
                offset += 1
                
                if type_marker == 0x20:  # String type
                    if offset >= len(binary_data):
                        break
                    value_len = binary_data[offset]
                    offset += 1
                    if offset + value_len <= len(binary_data):
                        value = binary_data[offset:offset + value_len].decode('utf-8', errors='ignore')
                        offset += value_len
                        record[field_name] = value
                        
                elif type_marker == 0x13:  # Float type (8 bytes)
                    if offset + 8 <= len(binary_data):
                        value = struct.unpack('>d', binary_data[offset:offset + 8])[0]
                        offset += 8
                        record[field_name] = value
                        
                elif type_marker == 0x10:  # Integer type
                    if offset < len(binary_data):
                        value = binary_data[offset]
                        offset += 1
                        record[field_name] = value
                        
                elif type_marker == 0x02 or type_marker == 0x03:  # Boolean type
                    record[field_name] = type_marker == 0x03
                    
                else:
                    # Unknown type, skip
                    offset += 1
            
            # Add record if it has data
            if record:
                records.append(record)
                
        except (IndexError, struct.error, UnicodeDecodeError):
            offset += 1
            continue
    
    # Filter records where 'available' is True
    filtered_records = [record for record in records if record.get('available') is True]
    
    return filtered_records