def deserialize_tpack_data(encoded_data):
    """
    Deserializes TPACK (Tagged PACKed) binary data from base64 encoding.

    TPACK format uses type tags followed by field names and values:
    - Type 1: boolean, Type 2: string, Type 16: int32, Type 19: float64
    - Type 48: start of record, Type 0: end of record/array

    Args:
        encoded_data (str): Base64 encoded TPACK data string

    Returns:
        dict: Deserialized data structure containing the parsed records
    """
    import base64
    import struct

    # Decode base64 data
    data = base64.b64decode(encoded_data)

    def read_string(data, pos):
        """Read a length-prefixed string from data at position pos"""
        if pos >= len(data):
            return "", pos
        
        length = data[pos]
        pos += 1
        
        if pos + length > len(data):
            return "", pos
            
        try:
            string_val = data[pos:pos+length].decode('utf-8')
            return string_val, pos + length
        except UnicodeDecodeError:
            # If we can't decode as UTF-8, return as raw bytes
            return data[pos:pos+length].hex(), pos + length

    def parse_value(data, pos, type_tag):
        """Parse a value based on its type tag"""
        if type_tag == 1:  # boolean
            return bool(data[pos]), pos + 1
        elif type_tag == 2:  # string
            return read_string(data, pos)
        elif type_tag == 16:  # int32
            if pos + 4 > len(data):
                return 0, pos
            value = struct.unpack('<I', data[pos:pos+4])[0]
            return value, pos + 4
        elif type_tag == 19:  # float64
            if pos + 8 > len(data):
                return 0.0, pos
            value = struct.unpack('<d', data[pos:pos+8])[0]
            return value, pos + 8
        else:
            # Skip unknown type
            return None, pos + 1

    def parse_record(data, pos):
        """Parse a single record starting at position pos"""
        record = {}

        while pos < len(data):
            if pos >= len(data):
                break
                
            type_tag = data[pos]
            pos += 1

            if type_tag == 0:  # End of record
                break
            elif type_tag == 48:  # Start of nested structure (0x30)
                field_name, pos = read_string(data, pos)
                nested_value, pos = parse_array(data, pos)
                record[field_name] = nested_value
            else:
                # Read field name
                field_name, pos = read_string(data, pos)
                # Parse value
                value, pos = parse_value(data, pos, type_tag)
                if field_name and value is not None:
                    record[field_name] = value

        return record, pos

    def parse_array(data, pos):
        """Parse an array of items"""
        items = []

        while pos < len(data):
            if pos >= len(data):
                break
                
            type_tag = data[pos]

            if type_tag == 0:  # End marker
                pos += 1
                break
            elif type_tag == 64:  # Start of record in array (0x40)
                pos += 1  # Skip the record start marker
                record, pos = parse_record(data, pos)
                items.append(record)
            else:
                # Single value or unknown tag, skip
                pos += 1

        return items, pos

    # Start parsing from the beginning
    result = {}
    pos = 0

    while pos < len(data):
        if pos >= len(data):
            break
            
        type_tag = data[pos]
        pos += 1

        if type_tag == 0:  # End marker
            break
        elif type_tag == 64:  # Start of record (0x40)
            # This seems to be the main record start
            while pos < len(data):
                if pos >= len(data):
                    break
                    
                inner_type = data[pos]
                pos += 1
                
                if inner_type == 0:
                    break
                elif inner_type == 48:  # Nested structure (0x30)
                    field_name, pos = read_string(data, pos)
                    nested_value, pos = parse_array(data, pos)
                    result[field_name] = nested_value
                else:
                    # Regular field
                    field_name, pos = read_string(data, pos)
                    value, pos = parse_value(data, pos, inner_type)
                    if field_name and value is not None:
                        result[field_name] = value
            break
        else:
            # Skip unknown tags
            continue

    return result