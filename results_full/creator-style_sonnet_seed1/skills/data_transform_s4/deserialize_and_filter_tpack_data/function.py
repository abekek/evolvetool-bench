def deserialize_and_filter_tpack_data(encoded_data):
    """
    Deserializes TPACK (msgpack) encoded data and filters records where 'available' is True.
    
    Utility: Decodes base64-encoded msgpack data, deserializes it, and returns only records 
    that have an 'available' field set to True. Handles nested structures and merges 
    availability information with product records.
    
    Args:
        encoded_data (str): Base64-encoded msgpack data containing product records
        
    Returns:
        list: List of dictionaries containing only records where 'available' is True
    """
    import base64
    import struct
    
    # Decode base64
    raw_data = base64.b64decode(encoded_data)
    
    # Simple msgpack deserializer for the expected format
    def deserialize_msgpack(data):
        pos = 0
        records = []
        
        while pos < len(data):
            if pos >= len(data):
                break
                
            # Read format byte
            format_byte = data[pos]
            pos += 1
            
            if format_byte == 0x85:  # fixmap with 5 elements (product record)
                record = {}
                for _ in range(5):
                    # Read key
                    if pos >= len(data):
                        break
                    key_format = data[pos]
                    pos += 1
                    
                    if key_format >= 0xa0 and key_format <= 0xbf:  # fixstr
                        key_len = key_format - 0xa0
                        key = data[pos:pos+key_len].decode('utf-8')
                        pos += key_len
                    else:
                        continue
                    
                    # Read value
                    if pos >= len(data):
                        break
                    value_format = data[pos]
                    pos += 1
                    
                    if value_format >= 0xa0 and value_format <= 0xbf:  # fixstr
                        value_len = value_format - 0xa0
                        value = data[pos:pos+value_len].decode('utf-8')
                        pos += value_len
                    elif value_format == 0xcb:  # float64
                        if pos + 8 <= len(data):
                            value = struct.unpack('>d', data[pos:pos+8])[0]
                            pos += 8
                        else:
                            break
                    elif value_format >= 0x00 and value_format <= 0x7f:  # positive fixint
                        value = value_format
                    elif value_format == 0xc3:  # true
                        value = True
                    elif value_format == 0xc2:  # false
                        value = False
                    else:
                        continue
                    
                    record[key] = value
                
                if record:
                    records.append(record)
            
            elif format_byte == 0x81:  # fixmap with 1 element (availability record)
                record = {}
                # Read key
                if pos >= len(data):
                    break
                key_format = data[pos]
                pos += 1
                
                if key_format >= 0xa0 and key_format <= 0xbf:  # fixstr
                    key_len = key_format - 0xa0
                    key = data[pos:pos+key_len].decode('utf-8')
                    pos += key_len
                    
                    # Read value
                    if pos < len(data):
                        value_format = data[pos]
                        pos += 1
                        
                        if value_format == 0xc3:  # true
                            value = True
                        elif value_format == 0xc2:  # false
                            value = False
                        else:
                            continue
                        
                        record[key] = value
                        records.append(record)
            else:
                # Skip unknown format
                continue
        
        return records
    
    # Deserialize the data
    all_records = deserialize_msgpack(raw_data)
    
    # Group records and merge availability information
    products = []
    current_product = None
    
    for record in all_records:
        if 'sku' in record:  # This is a product record
            current_product = record.copy()
            products.append(current_product)
        elif 'available' in record and current_product is not None:  # This is availability info
            current_product['available'] = record['available']
    
    # Filter records where available is True
    filtered_records = [record for record in products if record.get('available') == True]
    
    return filtered_records