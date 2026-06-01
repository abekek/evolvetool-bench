def deserialize_tpack(data: str) -> list[dict]:
    """
    Deserialize TPACK (binary packed) data format into structured records.
    
    TPACK format specification:
    - Header: 4-byte magic number (0x54504143 = 'TPAC'), 4-byte record count
    - Schema: 4-byte field count, then for each field: 1-byte type + null-terminated name
    - Records: Each record contains fields in schema order based on their types
    
    Field types:
    - 0x01: 32-bit signed integer (4 bytes)
    - 0x02: 64-bit double (8 bytes) 
    - 0x03: Boolean (1 byte, 0=False, 1=True)
    - 0x04: String (4-byte length + UTF-8 bytes)
    
    Parameters:
    data (str): Base64-encoded TPACK binary data
    
    Returns:
    list[dict]: List of records as dictionaries, or error info if parsing fails
    """
    import base64
    import struct
    
    try:
        # Decode base64 data
        try:
            binary_data = base64.b64decode(data)
        except Exception as e:
            return []
        
        if len(binary_data) < 8:
            return []
        
        offset = 0
        
        # Parse header
        magic, record_count = struct.unpack('<II', binary_data[offset:offset+8])
        offset += 8
        
        if magic != 0x54504143:  # 'TPAC' in little-endian
            return []
        
        if offset >= len(binary_data):
            return []
        
        # Parse schema
        field_count = struct.unpack('<I', binary_data[offset:offset+4])[0]
        offset += 4
        
        schema = []
        for _ in range(field_count):
            if offset >= len(binary_data):
                return []
            
            field_type = binary_data[offset]
            offset += 1
            
            # Read null-terminated field name
            name_start = offset
            while offset < len(binary_data) and binary_data[offset] != 0:
                offset += 1
            
            if offset >= len(binary_data):
                return []
            
            try:
                field_name = binary_data[name_start:offset].decode('utf-8')
            except UnicodeDecodeError:
                return []
            offset += 1  # Skip null terminator
            
            schema.append((field_type, field_name))
        
        # Parse records
        records = []
        for record_idx in range(record_count):
            record = {}
            
            for field_type, field_name in schema:
                if field_type == 0x01:  # 32-bit signed integer
                    if offset + 4 > len(binary_data):
                        return []
                    value = struct.unpack('<i', binary_data[offset:offset+4])[0]
                    offset += 4
                    
                elif field_type == 0x02:  # 64-bit double
                    if offset + 8 > len(binary_data):
                        return []
                    value = struct.unpack('<d', binary_data[offset:offset+8])[0]
                    offset += 8
                    
                elif field_type == 0x03:  # Boolean
                    if offset + 1 > len(binary_data):
                        return []
                    value = binary_data[offset] != 0
                    offset += 1
                    
                elif field_type == 0x04:  # String
                    if offset + 4 > len(binary_data):
                        return []
                    str_len = struct.unpack('<I', binary_data[offset:offset+4])[0]
                    offset += 4
                    
                    if offset + str_len > len(binary_data):
                        return []
                    try:
                        value = binary_data[offset:offset+str_len].decode('utf-8')
                    except UnicodeDecodeError:
                        return []
                    offset += str_len
                    
                else:
                    return []
                
                record[field_name] = value
            
            records.append(record)
        
        return records
        
    except Exception as e:
        return []