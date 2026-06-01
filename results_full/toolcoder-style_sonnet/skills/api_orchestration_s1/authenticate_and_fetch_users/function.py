def authenticate_and_fetch_users() -> list:
    """Authenticate with the API and return the list of users from the first page."""
    import urllib.request
    import urllib.parse
    import json
    import sys
    import traceback
    import base64
    
    try:
        # Step 1: Send GET request to /auth/info endpoint to discover the authentication scheme and requirements
        auth_info_url = "http://127.0.0.1:18080/auth/info"
        with urllib.request.urlopen(auth_info_url) as response:
            auth_info_data = response.read()
        
        # Step 2: Parse the authentication info response to determine auth method (API key, bearer token, basic auth, etc.)
        auth_info = json.loads(auth_info_data.decode('utf-8'))
        
        # Step 3: Generate the appropriate authentication header based on the discovered scheme
        headers = {}
        
        # Handle different authentication schemes
        if 'scheme' in auth_info:
            scheme = auth_info['scheme'].lower()
            
            if scheme == 'bearer':
                # Bearer token authentication
                token = auth_info.get('token', '')
                headers['Authorization'] = f'Bearer {token}'
            elif scheme == 'basic':
                # Basic authentication
                username = auth_info.get('username', '')
                password = auth_info.get('password', '')
                credentials = base64.b64encode(f'{username}:{password}'.encode('utf-8')).decode('utf-8')
                headers['Authorization'] = f'Basic {credentials}'
            elif scheme == 'apikey':
                # API key authentication
                api_key = auth_info.get('key', '')
                key_header = auth_info.get('header', 'X-API-Key')
                headers[key_header] = api_key
        elif 'token' in auth_info:
            # Default to bearer token if only token is provided
            headers['Authorization'] = f'Bearer {auth_info["token"]}'
        elif 'key' in auth_info:
            # Default to API key if only key is provided
            key_header = auth_info.get('header', 'X-API-Key')
            headers[key_header] = auth_info['key']
        
        # Step 4: Send authenticated GET request to /api/users endpoint with the generated auth header
        users_url = "http://127.0.0.1:18080/api/users"
        request = urllib.request.Request(users_url, headers=headers)
        
        with urllib.request.urlopen(request) as response:
            users_data = response.read()
        
        # Step 5: Parse the response JSON and extract the users list from the first page of results
        users_response = json.loads(users_data.decode('utf-8'))
        
        # Handle different response structures
        if isinstance(users_response, list):
            users_list = users_response
        elif isinstance(users_response, dict):
            # Check common keys for user lists in paginated responses
            users_list = users_response.get('users', 
                        users_response.get('data', 
                        users_response.get('results', [])))
        else:
            users_list = []
        
        # Step 6: Return the users list, handling any potential authentication or request errors
        return users_list
        
    except Exception as e:
        # Output full traceback to stderr for debugging
        sys.stderr.write(traceback.format_exc())
        return []