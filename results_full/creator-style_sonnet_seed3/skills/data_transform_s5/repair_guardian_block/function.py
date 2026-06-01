def repair_guardian_block(corrupted_data_b64):
    """
    Repair a corrupted GUARDIAN block using parity data and XOR reconstruction.

    Utility: Finds corrupted blocks by CRC mismatch, determines their parity groups,
    and uses XOR with parity blocks and valid blocks in the same group to reconstruct
    the original data. Verifies repair success by checking CRC matches.

    Args:
        corrupted_data_b64 (str): Base64 encoded GUARDIAN data with corrupted block(s)

    Returns:
        dict: Contains 'repaired_text' (UTF-8 decoded text), 'corrupted_blocks' 
              (list of block IDs), and 'repair_success' (boolean)
    """
    import base64
    import struct

    def crc16_ccitt(data):
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc

    # Decode base64 data
    data = base64.b64decode(corrupted_data_b64)

    # Parse header: GD + version(1) + block_size(1) + parity_group_size(1) + flags(1)
    header = struct.unpack('<2sBBBB', data[:6])
    magic, version, block_size, parity_group_size, flags = header

    # Parse blocks
    blocks = {}
    parity_blocks = {}
    offset = 6

    while offset < len(data):
        if offset + 6 > len(data):
            break

        # Parse block header: block_id(2) + data_length(1) + flags(1) + crc(2)
        # Total: 6 bytes for block header
        block_header = struct.unpack('<HBBH', data[offset:offset+6])
        block_id, data_length, block_flags, crc = block_header

        # Extract block data
        if offset + 6 + data_length > len(data):
            break
            
        block_data = data[offset+6:offset+6+data_length]

        # Check if this is a parity block (high bit of block_id or specific flag)
        if block_id >= 0xFF00:  # Parity blocks have high IDs
            parity_blocks[block_id] = {
                'data': block_data,
                'crc': crc,
                'length': data_length
            }
        else:  # Data block
            calculated_crc = crc16_ccitt(block_data)
            blocks[block_id] = {
                'data': block_data,
                'crc': crc,
                'length': data_length,
                'valid': calculated_crc == crc,
                'calculated_crc': calculated_crc
            }

        offset += 6 + data_length

    # Find corrupted blocks
    corrupted_blocks = [bid for bid, block in blocks.items() if not block['valid']]

    # Repair corrupted blocks
    for block_id in corrupted_blocks:
        parity_group = block_id % parity_group_size

        # Find parity block for this group
        parity_block_id = 0xFF00 + parity_group

        if parity_block_id not in parity_blocks:
            continue

        # Get all other blocks in the same parity group (excluding the corrupted one)
        group_blocks = [bid for bid in blocks.keys() 
                       if bid % parity_group_size == parity_group and bid != block_id]

        # Start with parity block data
        parity_data = parity_blocks[parity_block_id]['data']
        repair_data = bytearray(parity_data)

        # XOR with all other valid blocks in the same group
        for other_block_id in group_blocks:
            other_block = blocks[other_block_id]
            if other_block['valid']:
                other_data = other_block['data']
                # Ensure we don't go beyond the shorter array
                max_len = min(len(repair_data), len(other_data))
                for i in range(max_len):
                    repair_data[i] ^= other_data[i]
                
                # If other_data is longer, extend repair_data
                if len(other_data) > len(repair_data):
                    repair_data.extend(other_data[len(repair_data):])

        # Convert back to bytes
        repair_data = bytes(repair_data)

        # Verify repaired data matches expected CRC
        calculated_crc = crc16_ccitt(repair_data)
        if calculated_crc == blocks[block_id]['crc']:
            blocks[block_id]['data'] = repair_data
            blocks[block_id]['valid'] = True

    # Reconstruct full text from all valid data blocks (sorted by block ID)
    valid_data_blocks = [(bid, blocks[bid]) for bid in sorted(blocks.keys()) if blocks[bid]['valid']]
    full_data = b''.join(block['data'] for bid, block in valid_data_blocks)

    try:
        repaired_text = full_data.decode('utf-8', errors='replace')
    except:
        repaired_text = str(full_data)

    # Check if all blocks are now valid
    repair_success = all(blocks[bid]['valid'] for bid in blocks.keys())

    return {
        'repaired_text': repaired_text,
        'corrupted_blocks': corrupted_blocks,
        'repair_success': repair_success
    }