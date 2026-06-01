def decode_base64(encoded_string: str) -> bytes:
    """
    Decode base64 encoded binary data into bytes.
    
    Args:
        encoded_string: Base64 encoded string to decode
        
    Returns:
        bytes: The decoded binary data, or empty bytes if decoding fails
        
    Raises:
        TypeError: If input is not a string
        
    Note:
        Returns empty bytes on base64 decoding errors (invalid base64, etc.)
        but raises TypeError for invalid input types.
    """
    import base64
    
    # Validate input type
    if not isinstance(encoded_string, str):
        raise TypeError(f"Expected string, got {type(encoded_string).__name__}")
    
    try:
        # Remove any whitespace that might interfere with decoding
        cleaned_string = encoded_string.strip()
        
        # Decode the base64 string to bytes
        decoded_bytes = base64.b64decode(cleaned_string)
        
        return decoded_bytes
        
    except Exception:
        # Return empty bytes on any decoding error (invalid base64, etc.)
        return b''