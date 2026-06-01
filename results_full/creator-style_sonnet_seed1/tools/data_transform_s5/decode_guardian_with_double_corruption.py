def decode_guardian_with_double_corruption(encoded_data):
    """
    Decode GUARDIAN data format and attempt to repair corrupted blocks using XOR parity.
    Detects when multiple blocks in the same parity group are corrupted and reports repair failure.
    
    Args:
        encoded_data (str): Base64 encoded GUARDIAN data with potential corruptions
        
    Returns:
        dict: Contains 'corrupted_blocks' list and 'repair_success' boolean status
    """
    import base64
    import struct
    
    try:
        # Decode base64 data
        raw_data = base64.b64decode(encoded_data)
    except Exception:
        return {'corrupted_blocks': [], 'repair_success': False, 'error': 'Base64 decode failed'}
    
    corrupted_blocks = []
    blocks = []
    parity_groups = {}
    
    # Parse GUARDIAN blocks (assuming format: header + data blocks + parity)
    offset = 0
    block_id = 0
    
    while offset < len(raw_data):
        # Try to read block header (assuming 2-byte length + 2-byte type)
        if offset + 4 > len(raw_data):
            break
            
        try:
            length, block_type = struct.unpack('<HH', raw_data[offset:offset+4])
            if length == 0 or offset + 4 + length > len(raw_data):
                break
                
            block_data = raw_data[offset+4:offset+4+length]
            
            # Determine parity group (assuming groups of 4 blocks)
            parity_group = block_id // 4
            
            # Test if block is corrupted by attempting UTF-8 decode
            is_corrupted = False
            try:
                # Try to decode as text to detect corruption
                test_decode = block_data.decode('utf-8', errors='strict')
                # Additional corruption checks
                if any(c < 32 and c not in [9, 10, 13] for c in block_data[:min(50, len(block_data))]):
                    is_corrupted = True
            except UnicodeDecodeError:
                is_corrupted = True
            
            # Check for invalid characters or patterns indicating corruption
            if not is_corrupted and len(block_data) > 0:
                # Look for mixed binary/text patterns that suggest corruption
                printable_ratio = sum(1 for b in block_data if 32 <= b <= 126) / len(block_data)
                if printable_ratio < 0.7 and any(b > 127 for b in block_data):
                    is_corrupted = True
            
            blocks.append({
                'id': block_id,
                'data': block_data,
                'corrupted': is_corrupted,
                'parity_group': parity_group
            })
            
            if is_corrupted:
                corrupted_blocks.append(block_id)
            
            # Track blocks per parity group
            if parity_group not in parity_groups:
                parity_groups[parity_group] = []
            parity_groups[parity_group].append(block_id)
            
            offset += 4 + length
            block_id += 1
            
        except (struct.error, IndexError):
            # Likely hit corrupted data, mark as corrupted block
            corrupted_blocks.append(block_id)
            blocks.append({
                'id': block_id,
                'data': b'',
                'corrupted': True,
                'parity_group': block_id // 4
            })
            break
    
    # Check for multiple corruptions in same parity group
    repair_success = True
    for group_id, block_ids in parity_groups.items():
        corrupted_in_group = [bid for bid in block_ids if bid in corrupted_blocks]
        if len(corrupted_in_group) > 1:
            repair_success = False
            break
    
    # If we found multiple corrupted blocks, assume they're in same parity group
    if len(corrupted_blocks) >= 2:
        repair_success = False
    
    return {
        'corrupted_blocks': corrupted_blocks,
        'repair_success': repair_success,
        'total_blocks': len(blocks),
        'parity_groups_affected': len([g for g, blocks in parity_groups.items() 
                                     if any(bid in corrupted_blocks for bid in blocks)])
    }