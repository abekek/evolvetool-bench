def decode_guardian_data(base64_data):
    """
    Decode and verify GUARDIAN format data structure.
    
    Utility: Decodes base64 GUARDIAN data, extracts header information, parses blocks,
    reconstructs text content, and verifies data integrity using SHA-256 hash.
    
    Args:
        base64_data (str): Base64 encoded GUARDIAN format data
        
    Returns:
        dict: Contains 'text' (reconstructed message), 'blocks_count' (number of blocks),
        and 'integrity_valid' (boolean indicating hash verification result)
    """
    import base64
    import hashlib
    import struct
    
    # Decode base64 data
    binary_data = base64.b64decode(base64_data)
    
    # Parse header (first 44 bytes)
    magic = binary_data[:4]  # Should be b'GD\x01\x10'
    version = binary_data[4]
    blocks_count = struct.unpack('<I', binary_data[5:9])[0]  # Little-endian uint32
    stored_hash = binary_data[9:41]  # 32 bytes SHA-256 hash
    
    # Extract blocks data (everything after header)
    blocks_data = binary_data[41:]
    
    # Parse blocks
    blocks = []
    offset = 0
    
    for i in range(blocks_count):
        if offset >= len(blocks_data):
            break
            
        # Block header: ID (2 bytes) + Length (2 bytes, little-endian)
        block_id = struct.unpack('<H', blocks_data[offset:offset+2])[0]
        block_length = struct.unpack('<H', blocks_data[offset+2:offset+4])[0]
        
        # Block content
        content_start = offset + 4
        content_end = content_start + block_length
        content = blocks_data[content_start:content_end]
        
        blocks.append({
            'id': block_id,
            'length': block_length,
            'content': content
        })
        
        offset = content_end
    
    # Reconstruct text by concatenating block contents and removing null bytes
    text_parts = []
    for block in blocks:
        # Remove null bytes and decode as UTF-8, ignoring errors
        clean_content = block['content'].rstrip(b'\x00')
        try:
            text_part = clean_content.decode('utf-8', errors='ignore')
            text_parts.append(text_part)
        except:
            # If decode fails, convert to string representation
            text_parts.append(str(clean_content))
    
    reconstructed_text = ''.join(text_parts)
    
    # Verify integrity by computing SHA-256 of blocks data
    computed_hash = hashlib.sha256(blocks_data).digest()
    integrity_valid = computed_hash == stored_hash
    
    return {
        'text': reconstructed_text,
        'blocks_count': blocks_count,
        'integrity_valid': integrity_valid
    }