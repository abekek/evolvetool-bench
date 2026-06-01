def deserialize_tpack_data(base64_data):
    """
    Deserializes TPACK (Tagged Pack) binary data format from base64 encoded string.
    
    TPACK is a binary serialization format that uses type tags and variable-length integers.
    Supports maps, arrays, strings, integers of various sizes, and nested structures.
    
    Args:
        base64_data (str): Base64 encoded TPACK binary data
        
    Returns:
        dict: Deserialized data structure containing the original nested maps, arrays, and values
    """
    import base64
    import struct
    
    def read_varint(data, pos):
        """Read variable-length integer from data at position pos"""
        result = 0
        shift = 0
        while pos < len(data):
            byte = data[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
        return result, pos
    
    def read_signed_varint(data, pos):
        """Read signed variable-length integer (zigzag encoded)"""
        value, new_pos = read_varint(data, pos)
        # Zigzag decode: (n >> 1) ^ (-(n & 1))
        return (value >> 1) ^ (-(value & 1)), new_pos
    
    def parse_value(data, pos):
        """Parse a single value from TPACK data"""
        if pos >= len(data):
            return None, pos
            
        type_byte = data[pos]
        pos += 1
        
        # Map type (0x40 | count)
        if type_byte & 0x40:
            count = type_byte & 0x3F
            if count == 0x3F:  # Extended count
                count, pos = read_varint(data, pos)
            
            result = {}
            for _ in range(count):
                # Read key length and key
                key_len, pos = read_varint(data, pos)
                if pos + key_len > len(data):
                    break
                key = data[pos:pos + key_len].decode('utf-8')
                pos += key_len
                
                # Read value
                value, pos = parse_value(data, pos)
                result[key] = value
            return result, pos
        
        # Array type (0x30 | count)
        elif type_byte & 0x30 == 0x30:
            count = type_byte & 0x0F
            if count == 0x0F:  # Extended count
                count, pos = read_varint(data, pos)
            
            result = []
            for _ in range(count):
                value, pos = parse_value(data, pos)
                result.append(value)
            return result, pos
        
        # String type (0x20 | length)
        elif type_byte & 0x20:
            length = type_byte & 0x1F
            if length == 0x1F:  # Extended length
                length, pos = read_varint(data, pos)
            
            if pos + length > len(data):
                return "", len(data)
            result = data[pos:pos + length].decode('utf-8')
            return result, pos + length
        
        # Integer types
        elif type_byte & 0x10:
            int_type = type_byte & 0x0F
            if int_type == 0x00:  # uint8
                if pos >= len(data):
                    return 0, pos
                return data[pos], pos + 1
            elif int_type == 0x01:  # uint16
                if pos + 2 > len(data):
                    return 0, len(data)
                return struct.unpack('<H', data[pos:pos + 2])[0], pos + 2
            elif int_type == 0x02:  # uint32
                if pos + 4 > len(data):
                    return 0, len(data)
                return struct.unpack('<I', data[pos:pos + 4])[0], pos + 4
            elif int_type == 0x08:  # int8
                if pos >= len(data):
                    return 0, pos
                return struct.unpack('b', data[pos:pos + 1])[0], pos + 1
            elif int_type == 0x09:  # int32
                if pos + 4 > len(data):
                    return 0, len(data)
                return struct.unpack('<i', data[pos:pos + 4])[0], pos + 4
            elif int_type == 0x0A:  # varint
                return read_varint(data, pos)
            else:
                # Skip unknown integer type
                return 0, pos
        
        # Other types - skip for now
        else:
            return None, pos
    
    # Decode base64 data
    try:
        binary_data = base64.b64decode(base64_data)
    except Exception:
        return {"error": "Invalid base64 data"}
    
    # Parse the TPACK data
    try:
        result, _ = parse_value(binary_data, 0)
        return result if result is not None else {}
    except Exception as e:
        return {"error": f"Failed to parse TPACK data: {str(e)}"}