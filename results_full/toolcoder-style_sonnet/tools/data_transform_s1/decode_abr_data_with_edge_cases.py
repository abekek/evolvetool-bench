def decode_abr_data_with_edge_cases(base64_data: str) -> list[dict]:
    """Decode ABR format data handling embedded delimiters and encoding issues, returning JSON array of objects."""
    import base64
    import traceback
    import sys
    
    try:
        # Step 1: Decode base64 data to raw bytes without assuming UTF-8 encoding
        raw_bytes = base64.b64decode(base64_data)
        
        # Step 2: Parse the ABR header to determine record count and validate format structure
        if len(raw_bytes) < 2:
            return []
        
        # ABR format appears to start with a 2-byte header indicating record count
        record_count = int.from_bytes(raw_bytes[0:2], byteorder='big')
        data_bytes = raw_bytes[2:]
        
        # Step 3: Iterate through records using stateful parsing that tracks delimiter context
        records = []
        pos = 0
        
        for record_idx in range(record_count):
            record = {}
            
            # Parse fields until we hit the record delimiter (0xFF) or end of data
            while pos < len(data_bytes):
                # Look for field name (terminated by 0x00)
                field_name_start = pos
                field_name_end = pos
                
                # Find the null terminator for field name, being careful about embedded nulls
                while field_name_end < len(data_bytes) and data_bytes[field_name_end] != 0x00:
                    field_name_end += 1
                
                if field_name_end >= len(data_bytes):
                    break
                    
                field_name_bytes = data_bytes[field_name_start:field_name_end]
                pos = field_name_end + 1  # Skip the null terminator
                
                if pos >= len(data_bytes):
                    break
                
                # Look for field value (terminated by 0x00)
                field_value_start = pos
                field_value_end = pos
                
                # Find the null terminator for field value
                while field_value_end < len(data_bytes) and data_bytes[field_value_end] != 0x00:
                    field_value_end += 1
                
                if field_value_end >= len(data_bytes):
                    # Take remaining bytes as the value
                    field_value_bytes = data_bytes[field_value_start:]
                    pos = len(data_bytes)
                else:
                    field_value_bytes = data_bytes[field_value_start:field_value_end]
                    pos = field_value_end + 1  # Skip the null terminator
                
                # Step 4: Extract field name-value pairs for each record, applying defensive decoding
                field_name = self._safe_decode_bytes(field_name_bytes)
                field_value = self._safe_decode_bytes(field_value_bytes)
                
                # Step 5: Construct dictionary objects from parsed fields
                if field_name:  # Only add non-empty field names
                    record[field_name] = field_value
                
                # Check if we hit a record delimiter (0xFF)
                if pos < len(data_bytes) and data_bytes[pos] == 0xFF:
                    pos += 1  # Skip the record delimiter
                    break
            
            if record:  # Only add non-empty records
                records.append(record)
        
        # Step 6: Return the complete list of record dictionaries as the final JSON-serializable result
        return records
        
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        return []

def _safe_decode_bytes(self, byte_data: bytes) -> str:
    """Safely decode bytes to string using fallback encoding strategies."""
    if not byte_data:
        return ""
    
    # Try UTF-8 first
    try:
        return byte_data.decode('utf-8')
    except UnicodeDecodeError:
        pass
    
    # Try latin-1 as fallback (maps all byte values to unicode)
    try:
        return byte_data.decode('latin-1')
    except UnicodeDecodeError:
        pass
    
    # Last resort: decode with errors='replace'
    return byte_data.decode('utf-8', errors='replace')