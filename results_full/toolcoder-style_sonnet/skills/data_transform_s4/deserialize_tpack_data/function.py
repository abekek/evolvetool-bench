def deserialize_tpack_data(encoded_data: str) -> dict:
    """Deserialize TPACK encoded data into a structured dictionary representation."""
    import base64
    import struct
    import sys
    import traceback
    
    try:
        # Step 1: Decode the base64 encoded string to get the raw binary TPACK data
        binary_data = base64.b64decode(encoded_data)
        
        # Step 2: Parse the binary data according to TPACK format specifications, identifying field types and boundaries
        def parse_value(data, offset):
            if offset >= len(data):
                raise ValueError("Unexpected end of data")
            
            type_byte = data[offset]
            offset += 1
            
            if type_byte == 0x01:  # Boolean false
                return False, offset
            elif type_byte == 0x02:  # Boolean true
                return True, offset
            elif type_byte == 0x10:  # 32-bit integer
                if offset + 4 > len(data):
                    raise ValueError("Not enough data for 32-bit integer")
                value = struct.unpack('>I', data[offset:offset+4])[0]
                return value, offset + 4
            elif type_byte == 0x13:  # 64-bit float
                if offset + 8 > len(data):
                    raise ValueError("Not enough data for 64-bit float")
                value = struct.unpack('>d', data[offset:offset+8])[0]
                return value, offset + 8
            elif type_byte == 0x20:  # String
                if offset >= len(data):
                    raise ValueError("No length byte for string")
                length = data[offset]
                offset += 1
                if offset + length > len(data):
                    raise ValueError("Not enough data for string")
                value = data[offset:offset+length].decode('utf-8')
                return value, offset + length
            elif type_byte == 0x30:  # Array
                if offset >= len(data):
                    raise ValueError("No length byte for array")
                length = data[offset]
                offset += 1
                array = []
                for _ in range(length):
                    item, offset = parse_value(data, offset)
                    array.append(item)
                return array, offset
            elif type_byte == 0x40:  # Object
                if offset >= len(data):
                    raise ValueError("No field count for object")
                field_count = data[offset]
                offset += 1
                obj = {}
                for _ in range(field_count):
                    # Parse field name (string)
                    field_name, offset = parse_value(data, offset)
                    # Parse field value
                    field_value, offset = parse_value(data, offset)
                    obj[field_name] = field_value
                return obj, offset
            else:
                raise ValueError(f"Unknown type byte: 0x{type_byte:02x}")
        
        # Step 3: Extract and convert typed field values (strings, numbers, arrays, booleans) based on their type indicators
        # Step 4: Build the hierarchical data structure, handling nested objects like customer info and items array
        result, final_offset = parse_value(binary_data, 0)
        
        # Step 5: Validate the parsed data structure and handle any malformed or incomplete fields gracefully
        if not isinstance(result, dict):
            raise ValueError("Root value is not an object")
        
        # Validate expected structure
        expected_fields = ['order_id', 'customer', 'items', 'total', 'shipped', 'notes']
        for field in expected_fields:
            if field not in result:
                print(f"Warning: Expected field '{field}' not found", file=sys.stderr)
        
        # Validate customer structure if present
        if 'customer' in result and isinstance(result['customer'], dict):
            customer_fields = ['name', 'email']
            for field in customer_fields:
                if field not in result['customer']:
                    print(f"Warning: Expected customer field '{field}' not found", file=sys.stderr)
        
        # Validate items structure if present
        if 'items' in result and isinstance(result['items'], list):
            for i, item in enumerate(result['items']):
                if isinstance(item, dict):
                    item_fields = ['sku', 'qty', 'unit_price']
                    for field in item_fields:
                        if field not in item:
                            print(f"Warning: Expected item field '{field}' not found in item {i}", file=sys.stderr)
        
        # Step 6: Return the complete deserialized data as a properly structured dictionary
        return result
        
    except Exception as e:
        print(f"Error deserializing TPACK data: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return {}