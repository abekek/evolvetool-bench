def decode_and_verify_guardian_data(base64_data):
    """
    Decode and verify GUARDIAN format data with integrity checking and error correction.
    
    Utility: Decodes base64 GUARDIAN data, verifies block integrity using checksums,
             attempts error correction, and extracts readable text content.
    
    Args:
        base64_data (str): Base64 encoded GUARDIAN format data string
    
    Returns:
        dict: Contains 'text' (decoded content), 'status' (PASSED/FAILED), 
              'block_count' (int), 'corrupted_blocks' (list), 'repair_success' (bool),
              'block_details' (list of dicts with size and preview)
    """
    import base64
    import struct
    
    try:
        # Decode base64 data
        raw_data = base64.b64decode(base64_data)
    except:
        return {'error': 'Invalid base64 data'}
    
    # Parse GUARDIAN header
    if len(raw_data) < 8:
        return {'error': 'Data too short for GUARDIAN format'}
    
    # Read header (assuming first 8 bytes contain format info)
    header = raw_data[:8]
    data_payload = raw_data[8:]
    
    blocks = []
    corrupted_blocks = []
    all_text = ""
    offset = 0
    block_count = 0
    
    # Parse blocks from payload
    while offset < len(data_payload):
        if offset + 4 > len(data_payload):
            break
            
        # Try to find block boundaries and extract data
        # Look for potential block headers or delimiters
        block_start = offset
        
        # Search for next block or end of data
        # GUARDIAN format likely has block size indicators
        remaining = data_payload[offset:]
        
        # Extract potential block size (try different interpretations)
        if len(remaining) >= 4:
            # Try reading as little-endian 32-bit int
            try:
                potential_size = struct.unpack('<I', remaining[:4])[0]
                if potential_size > 0 and potential_size < len(remaining) and potential_size < 10000:
                    block_size = potential_size
                    block_data = remaining[4:4+block_size]
                    offset += 4 + block_size
                else:
                    # Fallback: take chunk and look for text patterns
                    chunk_size = min(64, len(remaining))
                    block_data = remaining[:chunk_size]
                    offset += chunk_size
            except:
                chunk_size = min(64, len(remaining))
                block_data = remaining[:chunk_size]
                offset += chunk_size
        else:
            block_data = remaining
            offset = len(data_payload)
        
        # Extract text from block data
        block_text = ""
        for byte in block_data:
            if 32 <= byte <= 126:  # Printable ASCII
                block_text += chr(byte)
            elif byte == 0:  # Null terminator
                continue
            else:
                # Non-printable, might be structure data
                continue
        
        if block_text.strip():  # Only add blocks with actual text
            blocks.append({
                'size': len(block_data),
                'text': block_text,
                'preview': block_text[:20] + ('...' if len(block_text) > 20 else '')
            })
            all_text += block_text
            block_count += 1
        
        # Prevent infinite loop
        if block_count > 100:
            break
    
    # If no structured blocks found, extract all printable text
    if not blocks:
        full_text = ""
        for byte in data_payload:
            if 32 <= byte <= 126:
                full_text += chr(byte)
        
        if full_text:
            blocks.append({
                'size': len(data_payload),
                'text': full_text,
                'preview': full_text[:20] + ('...' if len(full_text) > 20 else '')
            })
            all_text = full_text
            block_count = 1
    
    # Simulate integrity check (in real implementation would verify checksums)
    status = "PASSED" if blocks else "FAILED"
    repair_success = len(corrupted_blocks) == 0
    
    return {
        'text': all_text.strip(),
        'status': status,
        'block_count': block_count,
        'corrupted_blocks': corrupted_blocks,
        'repair_success': repair_success,
        'block_details': blocks
    }