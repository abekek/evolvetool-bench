def repair_guardian_block(corrupted_data_b64):
    """
    Repair a corrupted GUARDIAN block using parity data and XOR reconstruction.
    
    Utility: Analyzes GUARDIAN format data to find corrupted blocks (CRC mismatch),
    then uses XOR parity blocks to reconstruct the original data. Returns the
    fully repaired UTF-8 text along with repair status information.
    
    Args:
        corrupted_data_b64 (str): Base64 encoded GUARDIAN format data containing
                                 blocks with headers, data, and parity information
    
    Returns:
        dict: Contains 'repaired_text' (reconstructed UTF-8 string), 
              'corrupted_blocks' (list of corrupted block IDs), and 
              'repair_success' (boolean indicating if all repairs succeeded)
    """
    import base64
    import struct
    import zlib
    
    # Decode the base64 data
    data = base64.b64decode(corrupted_data_b64)
    
    # Parse GUARDIAN header (8 bytes)
    header = data[:8]
    magic, version, parity_group_size, num_blocks = struct.unpack('<4sBBH', header)
    
    # Parse all blocks
    blocks = {}
    parity_blocks = {}
    offset = 8
    
    for _ in range(num_blocks):
        # Parse block header (4 bytes)
        block_header = data[offset:offset+4]
        block_id, block_type, data_length = struct.unpack('<BBH', block_header)
        offset += 4
        
        # Read block data
        block_data = data[offset:offset+data_length]
        offset += data_length
        
        # Read CRC (4 bytes)
        crc_bytes = data[offset:offset+4]
        stored_crc = struct.unpack('<I', crc_bytes)[0]
        offset += 4
        
        # Calculate actual CRC
        calculated_crc = zlib.crc32(block_data) & 0xffffffff
        
        block_info = {
            'type': block_type,
            'data': block_data,
            'stored_crc': stored_crc,
            'calculated_crc': calculated_crc,
            'is_valid': stored_crc == calculated_crc
        }
        
        if block_type == 0:  # Data block
            blocks[block_id] = block_info
        elif block_type == 1:  # Parity block
            parity_blocks[block_id] = block_info
    
    # Find corrupted blocks
    corrupted_blocks = []
    for block_id, block_info in blocks.items():
        if not block_info['is_valid']:
            corrupted_blocks.append(block_id)
    
    # Repair corrupted blocks using parity
    repair_success = True
    
    for corrupted_id in corrupted_blocks:
        # Determine parity group
        parity_group = corrupted_id % parity_group_size
        
        # Find parity block for this group
        if parity_group not in parity_blocks:
            repair_success = False
            continue
            
        parity_data = parity_blocks[parity_group]['data']
        
        # Find all other valid blocks in the same parity group
        group_blocks = []
        for block_id, block_info in blocks.items():
            if block_id != corrupted_id and block_id % parity_group_size == parity_group:
                if block_info['is_valid']:
                    group_blocks.append(block_info['data'])
        
        # XOR parity with all other valid blocks to reconstruct
        reconstructed = bytearray(parity_data)
        for other_data in group_blocks:
            # Ensure same length by padding with zeros
            max_len = max(len(reconstructed), len(other_data))
            while len(reconstructed) < max_len:
                reconstructed.append(0)
            
            for i in range(len(other_data)):
                reconstructed[i] ^= other_data[i]
        
        # Update the corrupted block with reconstructed data
        blocks[corrupted_id]['data'] = bytes(reconstructed)
        
        # Verify the repair by checking CRC
        new_crc = zlib.crc32(blocks[corrupted_id]['data']) & 0xffffffff
        if new_crc == blocks[corrupted_id]['stored_crc']:
            blocks[corrupted_id]['is_valid'] = True
        else:
            repair_success = False
    
    # Reconstruct the full text from all data blocks
    repaired_text = ""
    sorted_blocks = sorted(blocks.items())
    
    for block_id, block_info in sorted_blocks:
        if block_info['type'] == 0:  # Data block
            try:
                # Decode as UTF-8, removing null padding
                text_data = block_info['data'].rstrip(b'\x00')
                repaired_text += text_data.decode('utf-8', errors='ignore')
            except:
                repair_success = False
    
    return {
        'repaired_text': repaired_text,
        'corrupted_blocks': corrupted_blocks,
        'repair_success': repair_success and len(corrupted_blocks) > 0
    }