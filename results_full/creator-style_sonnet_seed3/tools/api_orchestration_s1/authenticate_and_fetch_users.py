def authenticate_and_fetch_users(base_url="http://127.0.0.1:18080", username="", password=""):
    """
    Authenticate with a web service and fetch the first page of users.
    
    Utility: Performs HTTP authentication and retrieves user data from a REST API endpoint.
    This function handles the complete workflow of logging in and fetching user information.
    
    Args:
        base_url (str): The base URL of the web service (default: "http://127.0.0.1:18080")
        username (str): Username for authentication (empty string for anonymous/default auth)
        password (str): Password for authentication (empty string for anonymous/default auth)
    
    Returns:
        dict: A dictionary containing authentication status and user data, with keys:
            - 'auth_success': boolean indicating if authentication succeeded
            - 'users': list of user objects if successful, empty list if failed
            - 'error': error message if any step failed, None if successful
    """
    import urllib.request
    import urllib.parse
    import json
    import base64
    
    try:
        # Prepare authentication headers
        headers = {}
        if username or password:
            credentials = f"{username}:{password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers['Authorization'] = f'Basic {encoded_credentials}'
        
        # First, attempt authentication by accessing a common auth endpoint
        auth_endpoints = ['/api/auth', '/auth', '/login', '/api/users']
        auth_success = False
        
        for endpoint in auth_endpoints:
            try:
                auth_url = base_url.rstrip('/') + endpoint
                req = urllib.request.Request(auth_url, headers=headers)
                
                with urllib.request.urlopen(req) as response:
                    if response.getcode() == 200:
                        auth_success = True
                        # If this is already the users endpoint, parse and return
                        if 'users' in endpoint:
                            data = json.loads(response.read().decode())
                            return {
                                'auth_success': True,
                                'users': data if isinstance(data, list) else data.get('users', [data]),
                                'error': None
                            }
                        break
            except:
                continue
        
        # If authentication succeeded but we haven't fetched users yet, try users endpoints
        if auth_success:
            users_endpoints = ['/api/users', '/users', '/api/v1/users']
            
            for endpoint in users_endpoints:
                try:
                    users_url = base_url.rstrip('/') + endpoint
                    req = urllib.request.Request(users_url, headers=headers)
                    
                    with urllib.request.urlopen(req) as response:
                        if response.getcode() == 200:
                            data = json.loads(response.read().decode())
                            users_list = data if isinstance(data, list) else data.get('users', [data])
                            
                            return {
                                'auth_success': True,
                                'users': users_list,
                                'error': None
                            }
                except Exception as e:
                    continue
        
        # If no users endpoint worked, try fetching from root with pagination params
        try:
            root_url = base_url.rstrip('/') + '/?page=1&limit=10'
            req = urllib.request.Request(root_url, headers=headers)
            
            with urllib.request.urlopen(req) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode())
                    return {
                        'auth_success': True,
                        'users': data if isinstance(data, list) else [data],
                        'error': None
                    }
        except:
            pass
        
        return {
            'auth_success': auth_success,
            'users': [],
            'error': 'Could not fetch users data from any endpoint'
        }
        
    except Exception as e:
        return {
            'auth_success': False,
            'users': [],
            'error': f'Request failed: {str(e)}'
        }