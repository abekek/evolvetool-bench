def decode_base64(encoded_data: str) -> bytes:
    """
    Decode base64 encoded data into binary format.
    
    Args:
        encoded_data: Base64 encoded string to decode
        
    Returns:
        bytes: The decoded binary data, or empty bytes if decoding fails
        
    Note:
        Returns empty bytes on any decoding error (invalid base64, etc.)
    """
    import base64
    
    try:
        # Strip whitespace and validate input
        if not isinstance(encoded_data, str):
            return b''
            
        cleaned_data = encoded_data.strip()
        if not cleaned_data:
            return b''
            
        # Decode the base64 data
        decoded_bytes = base64.b64decode(cleaned_data, validate=True)
        return decoded_bytes
        
    except Exception:
        # Return empty bytes on any error (invalid base64, etc.)
        return b''