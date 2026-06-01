def decode_guardian_data(base64_data):
    """
    Decode GUARDIAN format data with error correction and integrity verification.
    
    Utility: Decodes base64-encoded GUARDIAN data blocks, extracts actual content while 
    handling padding, performs error correction using parity blocks, and verifies data integrity.
    
    Args:
        base64_data (str): Base64-encoded GUARDIAN format data
        
    Returns:
        dict: Contains 'decoded_text', 'integrity_status', 'blocks_processed', and 'corruption_details'
    """
    import base64
    import struct
    
    # Decode base64 data
    raw_data = base64.b64decode(base64_data)
    
    # Parse GUARDIAN header (first 4 bytes)
    if len(raw_data) < 4:
        return {'error': 'Invalid GUARDIAN data - too short'}
    
    magic = raw_data[:2]
    if magic != b'GD':
        return {'error': 'Invalid GUARDIAN magic bytes'}
    
    version = raw_data[2]
    block_size = raw_data[3]
    
    # Extract blocks
    blocks = []
    offset = 4
    block_count = 0
    
    while offset + block_size + 4 <= len(raw_data):
        # Read block header (4 bytes: data_len, checksum, flags, block_id)
        header = raw_data[offset:offset+4]
        data_len, checksum, flags, block_id = struct.unpack('BBBB', header)
        
        # Read block data
        block_data = raw_data[offset+4:offset+4+block_size]
        
        # Calculate checksum for verification
        calc_checksum = sum(block_data) & 0xFF
        
        blocks.append({
            'id': block_id,
            'data_len': data_len,
            'data': block_data,
            'checksum': checksum,
            'calc_checksum': calc_checksum,
            'flags': flags,
            'valid': checksum == calc_checksum
        })
        
        offset += 4 + block_size
        block_count += 1
    
    if not blocks:
        return {'error': 'No valid blocks found'}
    
    # Separate data blocks from parity blocks (flag bit 0 indicates parity)
    data_blocks = [b for b in blocks if not (b['flags'] & 1)]
    parity_blocks = [b for b in blocks if b['flags'] & 1]
    
    # Attempt error correction for corrupted blocks
    corrupted_blocks = []
    for block in data_blocks:
        if not block['valid']:
            corrupted_blocks.append(block['id'])
            # Try to correct using parity data
            if parity_blocks:
                parity_data = parity_blocks[0]['data']
                # Simple XOR correction (works for single block corruption)
                corrected_data = bytearray(block['data'])
                for i in range(min(len(corrected_data), len(parity_data))):
                    corrected_data[i] ^= parity_data[i]
                block['data'] = bytes(corrected_data)
                block['corrected'] = True
    
    # Extract actual text data (excluding padding)
    decoded_text = ""
    for block in sorted(data_blocks, key=lambda x: x['id']):
        actual_len = min(block['data_len'], len(block['data']))
        if actual_len > 0:
            # Extract only the actual data bytes, not padding
            text_data = block['data'][:actual_len]
            try:
                decoded_text += text_data.decode('utf-8', errors='ignore')
            except:
                # Fallback to latin-1 if UTF-8 fails
                decoded_text += text_data.decode('latin-1', errors='ignore')
    
    # Calculate integrity status
    total_blocks = len(data_blocks)
    valid_blocks = sum(1 for b in data_blocks if b['valid'] or b.get('corrected', False))
    corruption_detected = len(corrupted_blocks) > 0
    
    return {
        'decoded_text': decoded_text,
        'integrity_status': 'Clean' if not corruption_detected else f'Corruption detected in {len(corrupted_blocks)} blocks',
        'blocks_processed': total_blocks,
        'corruption_details': corrupted_blocks,
        'total_blocks': block_count,
        'parity_blocks': len(parity_blocks),
        'version': version,
        'block_size': block_size
    }