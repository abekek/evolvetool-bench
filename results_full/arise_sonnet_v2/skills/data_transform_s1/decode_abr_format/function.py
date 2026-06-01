def decode_abr_format(abr_data: str) -> list[dict]:
    """
    Decode ARISE Binary Record (ABR) format data from base64 encoded binary.
    
    The ABR format appears to use:
    - 0xFF as record separator
    - Length-prefixed strings where first byte is string length
    - Key-value pairs within each record
    
    Args:
        abr_data: Base64 encoded ABR format binary data
        
    Returns:
        List of dictionaries containing the decoded records, or error info if decoding fails
    """
    import base64
    
    try:
        # Handle empty input
        if not abr_data.strip():
            return []
            
        # Decode base64 to binary
        binary_data = base64.b64decode(abr_data)
        
        if not binary_data:
            return []
        
        records = []
        pos = 0
        
        while pos < len(binary_data):
            # Skip any leading 0xFF separators
            while pos < len(binary_data) and binary_data[pos] == 0xFF:
                pos += 1
            
            if pos >= len(binary_data):
                break
                
            record = {}
            
            # Parse key-value pairs until we hit 0xFF or end of data
            while pos < len(binary_data) and binary_data[pos] != 0xFF:
                # Read key length and key
                if pos >= len(binary_data):
                    break
                    
                key_len = binary_data[pos]
                pos += 1
                
                if pos + key_len > len(binary_data):
                    break
                    
                key = binary_data[pos:pos + key_len].decode('utf-8', errors='replace')
                pos += key_len
                
                # Read value length and value
                if pos >= len(binary_data):
                    break
                    
                value_len = binary_data[pos]
                pos += 1
                
                if pos + value_len > len(binary_data):
                    break
                    
                value = binary_data[pos:pos + value_len].decode('utf-8', errors='replace')
                pos += value_len
                
                record[key] = value
            
            if record:  # Only add non-empty records
                records.append(record)
        
        return records
        
    except Exception as e:
        return [{"error": f"Failed to decode ABR format: {str(e)}"}]