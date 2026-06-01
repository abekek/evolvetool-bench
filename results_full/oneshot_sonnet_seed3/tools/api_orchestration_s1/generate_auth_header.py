def generate_auth_header(auth_scheme, username='', password='', token='', api_key=''):
    """
    Generate Authorization header for different auth schemes.
    Supports Basic, Bearer, and API-Key authentication.
    """
    import base64
    
    auth_scheme = auth_scheme.lower().strip()
    
    if auth_scheme == 'basic':
        if not username:
            return 'Basic '
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('ascii')
        return f"Basic {encoded_credentials}"
    
    elif auth_scheme == 'bearer':
        if not token:
            return 'Bearer '
        return f"Bearer {token}"
    
    elif auth_scheme in ['api-key', 'apikey', 'x-api-key']:
        if not api_key:
            return 'API-Key '
        return f"API-Key {api_key}"
    
    elif auth_scheme == 'token':
        if not token:
            return 'Token '
        return f"Token {token}"
    
    else:
        # Generic format for unknown schemes
        credential = token or api_key or f"{username}:{password}"
        return f"{auth_scheme.title()} {credential}"
