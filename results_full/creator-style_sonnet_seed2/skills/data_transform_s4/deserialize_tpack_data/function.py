def deserialize_tpack_data(base64_data):
    """
    Deserialize TPACK (Tag-Pack) binary data format from base64 encoded string.
    
    TPACK is a binary serialization format that stores data as type-length-value tuples.
    This function decodes the base64 data and parses the binary structure to extract
    the original data fields and values.
    
    Args:
        base64_data (str): Base64 encoded TPACK binary data
        
    Returns:
        dict: Deserialized data structure with all fields and values
    """
    import base64
    import struct
    
    # Decode base64 to binary
    binary_data = base64.b64decode(base64_data)
    
    def parse_value(data, offset, value_type):
        """Parse different data types from binary data"""
        if value_type == 1:  # String
            length = data[offset]
            offset += 1
            value = data[offset:offset + length].decode('utf-8')
            return value, offset + length
        elif value_type == 2:  # Boolean
            value = bool(data[offset])
            return value, offset + 1
        elif value_type == 3:  # Integer (varint)
            value = 0
            shift = 0
            while offset < len(data):
                byte = data[offset]
                offset += 1
                value |= (byte & 0x7F) << shift
                if (byte & 0x80) == 0:
                    break
                shift += 7
            return value, offset
        elif value_type == 19:  # Double (8 bytes)
            value = struct.unpack('>d', data[offset:offset + 8])[0]
            return value, offset + 8
        elif value_type == 16:  # 32-bit integer
            value = struct.unpack('>I', data[offset:offset + 4])[0]
            return value, offset + 4
        elif value_type == 48:  # Array/List
            return parse_array(data, offset)
        else:
            # Default: treat as string with length prefix
            if offset < len(data):
                length = data[offset]
                offset += 1
                if offset + length <= len(data):
                    value = data[offset:offset + length].decode('utf-8', errors='ignore')
                    return value, offset + length
            return None, offset
    
    def parse_array(data, offset):
        """Parse array structure"""
        items = []
        # Arrays can contain multiple items, parse until we hit a different type marker
        while offset < len(data):
            # Check if next byte looks like a new field type
            if data[offset] == 64:  # Start of new object
                offset += 1
                item, offset = parse_object(data, offset)
                items.append(item)
            else:
                break
        return items, offset
    
    def parse_object(data, offset):
        """Parse object structure"""
        obj = {}
        while offset < len(data):
            if offset >= len(data):
                break
                
            # Read field type
            field_type = data[offset]
            offset += 1
            
            if field_type == 64:  # End of current object, start of new one
                offset -= 1  # Back up to let parent handle this
                break
            elif field_type == 0:  # End marker
                break
                
            # Read field name length and name
            if offset >= len(data):
                break
            name_length = data[offset]
            offset += 1
            
            if offset + name_length > len(data):
                break
                
            field_name = data[offset:offset + name_length].decode('utf-8')
            offset += name_length
            
            # Parse field value based on type
            value, offset = parse_value(data, offset, field_type)
            obj[field_name] = value
            
        return obj, offset
    
    # Start parsing from the beginning
    result, _ = parse_object(binary_data, 0)
    return result