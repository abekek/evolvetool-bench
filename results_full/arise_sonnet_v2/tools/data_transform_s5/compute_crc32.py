def compute_crc32(data: bytes) -> int:
    """
    Compute CRC32 checksum for data integrity verification.
    
    Args:
        data: The bytes data to compute CRC32 checksum for
        
    Returns:
        int: CRC32 checksum as an unsigned 32-bit integer, or -1 if error occurred
    """
    import struct
    
    try:
        if not isinstance(data, bytes):
            return -1
            
        # CRC32 polynomial (IEEE 802.3)
        polynomial = 0xEDB88320
        
        # Initialize CRC table
        crc_table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ polynomial
                else:
                    crc >>= 1
            crc_table.append(crc)
        
        # Compute CRC32
        crc = 0xFFFFFFFF
        for byte in data:
            crc = crc_table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
        
        # Return final CRC32 as unsigned 32-bit integer
        return crc ^ 0xFFFFFFFF
        
    except Exception:
        return -1