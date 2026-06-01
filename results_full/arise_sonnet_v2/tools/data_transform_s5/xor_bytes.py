def xor_bytes(data1: bytes, data2: bytes) -> bytes:
    """
    Perform XOR operation on two byte sequences for parity-based error correction.
    
    This function XORs two byte sequences element-wise, which is commonly used in
    error correction schemes like RAID parity calculations and data recovery operations.
    If the input sequences have different lengths, the operation is performed up to
    the length of the shorter sequence.
    
    Args:
        data1 (bytes): First byte sequence
        data2 (bytes): Second byte sequence
        
    Returns:
        bytes: Result of XOR operation between data1 and data2, or empty bytes if error
    """
    try:
        if not isinstance(data1, bytes) or not isinstance(data2, bytes):
            return b''
        
        # Handle empty inputs
        if len(data1) == 0 or len(data2) == 0:
            return b''
        
        # XOR up to the length of the shorter sequence
        min_length = min(len(data1), len(data2))
        result = bytearray()
        
        for i in range(min_length):
            result.append(data1[i] ^ data2[i])
        
        return bytes(result)
        
    except Exception:
        return b''