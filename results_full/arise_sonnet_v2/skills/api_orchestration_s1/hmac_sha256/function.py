def hmac_sha256(secret: str, message: str) -> str:
    """
    Generate HMAC-SHA256 hash for authentication and security purposes.
    
    Args:
        secret: The secret key used for HMAC generation
        message: The message to be authenticated
    
    Returns:
        Hexadecimal string representation of the HMAC-SHA256 hash, or error message if generation fails
    """
    import hmac
    import hashlib
    
    try:
        # Handle None inputs explicitly
        if secret is None or message is None:
            return "Error generating HMAC-SHA256: Input cannot be None"
        
        # Convert strings to bytes if they aren't already
        if isinstance(secret, str):
            secret_bytes = secret.encode('utf-8')
        elif isinstance(secret, bytes):
            secret_bytes = secret
        else:
            return f"Error generating HMAC-SHA256: Invalid secret type {type(secret)}"
            
        if isinstance(message, str):
            message_bytes = message.encode('utf-8')
        elif isinstance(message, bytes):
            message_bytes = message
        else:
            return f"Error generating HMAC-SHA256: Invalid message type {type(message)}"
        
        # Generate HMAC-SHA256
        hmac_hash = hmac.new(secret_bytes, message_bytes, hashlib.sha256)
        
        # Return hexadecimal representation
        return hmac_hash.hexdigest()
        
    except Exception as e:
        return f"Error generating HMAC-SHA256: {str(e)}"