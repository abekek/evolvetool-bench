def deserialize_tpack_and_filter_available(base64_data):
    """
    Deserializes TPACK data from base64 and filters records where 'available' is True.
    
    Utility: Decodes base64 TPACK data, parses the binary format to extract product records,
    and returns only those records where the 'available' field is True.
    
    Args:
        base64_data (str): Base64 encoded TPACK data containing product records
        
    Returns:
        list: List of dictionaries containing filtered product records with keys:
              'sku', 'name', 'price', 'qty', 'available'
    """
    import base64
    import struct
    
    # Decode base64 data
    binary_data = base64.b64decode(base64_data)
    
    records = []
    offset = 0
    
    while offset < len(binary_data):
        try:
            record = {}
            
            # Read record marker (0x03)
            if offset >= len(binary_data) or binary_data[offset] != 0x03:
                break
            offset += 1
            
            # Parse fields in the record
            while offset < len(binary_data):
                # Read field name length
                if offset >= len(binary_data):
                    break
                    
                field_name_len = binary_data[offset]
                offset += 1
                
                # Read field name
                if offset + field_name_len > len(binary_data):
                    break
                    
                field_name = binary_data[offset:offset + field_name_len].decode('utf-8')
                offset += field_name_len
                
                # Read field type/value length
                if offset >= len(binary_data):
                    break
                    
                value_len = binary_data[offset]
                offset += 1
                
                # Read and parse field value based on field name
                if field_name in ['sku', 'name']:
                    if offset + value_len > len(binary_data):
                        break
                    record[field_name] = binary_data[offset:offset + value_len].decode('utf-8')
                    offset += value_len
                elif field_name == 'price':
                    if offset + 8 > len(binary_data):
                        break
                    record[field_name] = struct.unpack('>d', binary_data[offset:offset + 8])[0]
                    offset += 8
                elif field_name == 'qty':
                    if offset + 1 > len(binary_data):
                        break
                    record[field_name] = binary_data[offset]
                    offset += 1
                elif field_name == 'available':
                    record[field_name] = True
                    # Available field seems to be just a marker
                    break
                else:
                    # Skip unknown field
                    if offset + value_len > len(binary_data):
                        break
                    offset += value_len
            
            # Add record if it has the required fields
            if 'sku' in record and 'available' in record:
                records.append(record)
                
        except (struct.error, UnicodeDecodeError, IndexError):
            break
    
    # Filter records where available is True
    filtered_records = [record for record in records if record.get('available', False)]
    
    return filtered_records