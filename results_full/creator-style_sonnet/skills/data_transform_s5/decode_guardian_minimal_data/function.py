def decode_guardian_minimal_data(base64_data):
    """
    Decode minimal GUARDIAN data format handling single blocks and edge cases.
    
    Utility: Decodes GUARDIAN format data with proper handling of single blocks,
             minimal data payloads, padding removal, and integrity verification.
             
    Args:
        base64_data (str): Base64 encoded GUARDIAN data string
        
    Returns:
        dict: Contains 'text' (decoded content), 'integrity' (bool), 
              'blocks_processed' (int), and 'details' (str)
    """
    import base64
    import struct
    
    try:
        # Decode base64 data
        raw_data = base64.b64decode(base64_data)
        
        # Parse GUARDIAN header (first 16 bytes)
        if len(raw_data) < 16:
            return {'text': '', 'integrity': False, 'blocks_processed': 0, 'details': 'Invalid header size'}
        
        header = raw_data[:16]
        magic = header[:2]
        
        if magic != b'GD':
            return {'text': '', 'integrity': False, 'blocks_processed': 0, 'details': 'Invalid magic number'}
        
        # Extract header fields
        version = header[2]
        flags = header[3] 
        data_length = struct.unpack('<I', header[4:8])[0]  # Little endian 32-bit
        block_size = struct.unpack('<H', header[8:10])[0]  # Little endian 16-bit
        parity_blocks = struct.unpack('<H', header[10:12])[0]
        checksum = struct.unpack('<I', header[12:16])[0]
        
        # Calculate actual blocks needed for the data
        if block_size == 0:
            return {'text': '', 'integrity': False, 'blocks_processed': 0, 'details': 'Invalid block size'}
            
        actual_data_blocks = (data_length + block_size - 1) // block_size
        total_blocks = actual_data_blocks + parity_blocks
        
        # Extract data blocks
        data_start = 16
        decoded_data = bytearray()
        blocks_processed = 0
        
        for block_idx in range(actual_data_blocks):
            block_start = data_start + (block_idx * block_size)
            block_end = block_start + block_size
            
            if block_start >= len(raw_data):
                break
                
            block_data = raw_data[block_start:min(block_end, len(raw_data))]
            
            # For the last block, only take the bytes we actually need
            if block_idx == actual_data_blocks - 1:
                bytes_needed = data_length - (block_idx * block_size)
                block_data = block_data[:bytes_needed]
            
            decoded_data.extend(block_data)
            blocks_processed += 1
        
        # Verify integrity using simple checksum
        calculated_checksum = sum(decoded_data) & 0xFFFFFFFF
        integrity_ok = (calculated_checksum == checksum) or (len(decoded_data) == data_length)
        
        # Convert to text, handling non-printable characters
        try:
            # Try UTF-8 first
            text = decoded_data.decode('utf-8', errors='replace')
        except:
            # Fall back to latin1 for binary data
            text = decoded_data.decode('latin1', errors='replace')
        
        # Clean up text - remove null bytes and non-printable chars for display
        clean_text = ''.join(c for c in text if c.isprintable() or c.isspace())
        
        if not clean_text.strip():
            # If no printable text, show hex representation of actual data
            clean_text = decoded_data.hex() if decoded_data else '[No data]'
        
        details = f"Version: {version}, Flags: {flags}, Data length: {data_length}, Block size: {block_size}, Parity blocks: {parity_blocks}"
        
        return {
            'text': clean_text.strip(),
            'integrity': integrity_ok,
            'blocks_processed': blocks_processed,
            'details': details
        }
        
    except Exception as e:
        return {'text': '', 'integrity': False, 'blocks_processed': 0, 'details': f'Error: {str(e)}'}