def parse_binary_data(data: bytes, format_spec: str) -> list:
    """
    Parse binary data using struct format specifications to extract structured fields.
    
    Args:
        data: Binary data to parse
        format_spec: Struct format string (e.g., '>I' for big-endian uint32, '<HH' for two little-endian uint16s)
                    Can also be a comma-separated list of formats for sequential parsing
    
    Returns:
        List of parsed values, or list with single dict containing error info if parsing fails
    
    Format specification examples:
        - '>I': Big-endian unsigned int (4 bytes)
        - '<H': Little-endian unsigned short (2 bytes)
        - 'B': Unsigned char (1 byte)
        - '>IH,B': Parse big-endian uint32 + uint16, then unsigned char (comma separates sequential operations)
    """
    import struct
    
    if not isinstance(data, bytes):
        return [{"error": "Input data must be bytes"}]
    
    if not format_spec:
        return [{"error": "Format specification cannot be empty"}]
    
    try:
        results = []
        offset = 0
        
        # Split format_spec by comma for sequential parsing
        format_parts = [part.strip() for part in format_spec.split(',')]
        
        for fmt in format_parts:
            if not fmt:
                continue
                
            try:
                # Calculate size needed for this format
                size = struct.calcsize(fmt)
                
                if offset + size > len(data):
                    return [{"error": f"Not enough data: need {size} bytes at offset {offset}, but only {len(data) - offset} bytes remaining"}]
                
                # Extract the chunk of data for this format
                chunk = data[offset:offset + size]
                
                # Unpack the data
                unpacked = struct.unpack(fmt, chunk)
                
                # If single value, append directly; if multiple, extend the list
                if len(unpacked) == 1:
                    results.append(unpacked[0])
                else:
                    results.extend(unpacked)
                
                offset += size
                
            except struct.error as e:
                return [{"error": f"Struct format error: {str(e)}"}]
        
        return results
        
    except Exception as e:
        return [{"error": f"Unexpected error: {str(e)}"}]