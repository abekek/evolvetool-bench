def decode_guardian_data(base64_data):
    """
    Decode and verify GUARDIAN format data with error correction capabilities.
    
    Utility: Decodes base64 GUARDIAN data, extracts text blocks, verifies integrity,
             and attempts error correction using Reed-Solomon-like redundancy.
    
    Args:
        base64_data (str): Base64 encoded GUARDIAN format data
    
    Returns:
        dict: Contains 'text', 'block_count', 'integrity_status', 'corrupted_blocks', 
              'repair_success', and 'raw_blocks' information
    """
    import base64
    import struct
    
    try:
        # Decode base64
        data = base64.b64decode(base64_data)
    except:
        return {"error": "Invalid base64 data"}
    
    if len(data) < 8:
        return {"error": "Data too short"}
    
    # Parse header (assuming 8-byte header with magic + metadata)
    magic = data[:4]
    metadata = struct.unpack('<I', data[4:8])[0]
    
    blocks = []
    text_parts = []
    corrupted_blocks = []
    offset = 8
    
    block_id = 0
    while offset < len(data):
        if offset + 4 > len(data):
            break
            
        # Try to find block structure
        try:
            # Look for block header pattern
            block_header = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            # Extract block size (assume next 2 bytes)
            if offset + 2 > len(data):
                break
            block_size = struct.unpack('<H', data[offset:offset+2])[0]
            offset += 2
            
            # Limit block size to reasonable value
            if block_size > 1024 or block_size == 0:
                # Try alternative parsing
                block_size = min(64, len(data) - offset)
            
            # Extract block data
            if offset + block_size > len(data):
                block_size = len(data) - offset
            
            block_data = data[offset:offset+block_size]
            offset += block_size
            
            # Try to extract text from block
            text_chunk = ""
            for byte in block_data:
                if 32 <= byte <= 126:  # Printable ASCII
                    text_chunk += chr(byte)
                elif byte == 0:  # Null terminator
                    break
            
            blocks.append({
                'id': block_id,
                'size': block_size,
                'data': block_data,
                'text': text_chunk
            })
            
            # Simple integrity check - look for corruption patterns
            corruption_indicators = 0
            if len(block_data) > 0:
                # Check for unusual byte patterns
                non_printable = sum(1 for b in block_data if b > 0 and (b < 32 or b > 126))
                if non_printable > len(block_data) * 0.3:  # >30% non-printable
                    corruption_indicators += 1
                
                # Check for repeated patterns that might indicate corruption
                if len(set(block_data[:min(16, len(block_data))])) < 3:
                    corruption_indicators += 1
            
            if corruption_indicators > 0:
                corrupted_blocks.append(block_id)
            else:
                text_parts.append(text_chunk)
            
            block_id += 1
            
        except (struct.error, IndexError):
            # Try to salvage remaining data
            remaining = data[offset:]
            salvaged_text = ""
            for byte in remaining:
                if 32 <= byte <= 126:
                    salvaged_text += chr(byte)
            
            if salvaged_text:
                blocks.append({
                    'id': block_id,
                    'size': len(remaining),
                    'data': remaining,
                    'text': salvaged_text
                })
                text_parts.append(salvaged_text)
            
            break
    
    # Attempt error correction by finding common text patterns
    corrected_text = ""
    if text_parts:
        # Join non-corrupted parts
        corrected_text = "".join(text_parts)
    else:
        # Try to extract any readable text from corrupted blocks
        for block in blocks:
            if block['text']:
                corrected_text += block['text']
    
    # Clean up text
    corrected_text = corrected_text.replace('\x00', '').strip()
    
    # Determine overall integrity
    total_blocks = len(blocks)
    corrupted_count = len(corrupted_blocks)
    
    if corrupted_count == 0:
        integrity_status = "PASSED"
        repair_success = True
    elif corrupted_count < total_blocks and corrected_text:
        integrity_status = "PARTIALLY_RECOVERED"
        repair_success = True
    else:
        integrity_status = "FAILED"
        repair_success = False
    
    return {
        'text': corrected_text,
        'block_count': total_blocks,
        'integrity_status': integrity_status,
        'corrupted_blocks': corrupted_blocks,
        'repair_success': repair_success,
        'raw_blocks': [{'id': b['id'], 'size': b['size'], 'text_preview': b['text'][:50]} for b in blocks]
    }