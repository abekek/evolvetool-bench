def decode_guardian_format(base64_data):
    """
    Decode and verify GUARDIAN (Guarded Data Integrity Archive) block format data.
    
    Utility: Parses GUARDIAN format binary data, validates CRC-16 checksums and XOR parity,
    extracts data blocks, and reconstructs the original UTF-8 text content.
    
    Args:
        base64_data (str): Base64 encoded GUARDIAN format binary data
        
    Returns:
        dict: Contains 'text' (decoded UTF-8 string), 'blocks' (total data blocks count),
              and 'integrity' (list of validation results per block with block_id, 
              crc_valid, and parity_valid flags)
    """
    import base64
    import struct
    
    def crc16_ccitt(data, init=0xFFFF, poly=0x1021):
        """Calculate CRC-16/CCITT checksum"""
        crc = init
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc
    
    # Decode base64
    data = base64.b64decode(base64_data)
    
    # Parse header (6 bytes)
    magic = struct.unpack('>H', data[0:2])[0]
    version = data[2]
    block_size = data[3]
    parity_group_size = data[4] 
    total_data_blocks = data[5]
    
    if magic != 0x4744:  # 'GD'
        raise ValueError("Invalid GUARDIAN magic number")
    
    # Parse data blocks
    offset = 6
    blocks = {}
    integrity_results = []
    
    # Read data blocks
    for _ in range(total_data_blocks):
        block_id = struct.unpack('>H', data[offset:offset+2])[0]
        data_length = data[offset+2]
        
        # Extract unpadded data
        block_data = data[offset+3:offset+3+data_length]
        
        # Skip to checksum (after padded data)
        checksum_offset = offset + 3 + block_size
        crc_checksum = struct.unpack('>H', data[checksum_offset:checksum_offset+2])[0]
        xor_parity = data[checksum_offset+2]
        
        # Verify CRC-16 on unpadded data
        calculated_crc = crc16_ccitt(block_data)
        crc_valid = (calculated_crc == crc_checksum)
        
        # Verify XOR parity on unpadded data
        calculated_parity = 0
        for byte in block_data:
            calculated_parity ^= byte
        parity_valid = (calculated_parity == xor_parity)
        
        blocks[block_id] = block_data
        integrity_results.append({
            'block_id': block_id,
            'crc_valid': crc_valid,
            'parity_valid': parity_valid
        })
        
        offset += 3 + block_size + 3  # block header + padded data + checksum + parity
    
    # Reconstruct text from blocks in order
    text_parts = []
    for block_id in sorted(blocks.keys()):
        text_parts.append(blocks[block_id].decode('utf-8'))
    
    reconstructed_text = ''.join(text_parts)
    
    return {
        'text': reconstructed_text,
        'blocks': total_data_blocks,
        'integrity': integrity_results
    }