def authenticate_and_fetch_users(base_url="http://127.0.0.1:18080", username="", password="", page=1, per_page=10):
    """
    Utility: Authenticates with a web service and fetches the first page of users
    
    Args:
        base_url (str): The base URL of the API service (default: "http://127.0.0.1:18080")
        username (str): Username for authentication (default: empty string for auto-detection)
        password (str): Password for authentication (default: empty string for auto-detection)
        page (int): Page number to fetch (default: 1)
        per_page (int): Number of users per page (default: 10)
    
    Returns:
        dict: Response containing authentication status and user data, or error information
    """
    import urllib.request
    import urllib.parse
    import urllib.error
    import json
    import base64
    
    try:
        # First, try to get authentication requirements or attempt common endpoints
        auth_endpoints = ['/auth/login', '/login', '/api/auth', '/authenticate']
        users_endpoints = ['/users', '/api/users', '/api/v1/users']
        
        # Try basic authentication first if credentials provided
        if username and password:
            # Create basic auth header
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers = {'Authorization': f'Basic {credentials}'}
        else:
            headers = {}
        
        # Add common headers
        headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Python-Auth-Client/1.0'
        })
        
        # Try to fetch users directly first (in case no auth required or basic auth works)
        for endpoint in users_endpoints:
            try:
                url = f"{base_url.rstrip('/')}{endpoint}"
                params = urllib.parse.urlencode({'page': page, 'per_page': per_page})
                full_url = f"{url}?{params}"
                
                req = urllib.request.Request(full_url, headers=headers)
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode())
                        return {
                            'success': True,
                            'method': 'direct_access' if not username else 'basic_auth',
                            'endpoint': endpoint,
                            'data': data,
                            'page': page,
                            'per_page': per_page
                        }
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    continue  # Try next endpoint or auth method
                elif e.code == 404:
                    continue  # Try next endpoint
                else:
                    # Some other error, but continue trying
                    continue
            except Exception:
                continue
        
        # If direct access failed, try authentication endpoints
        for auth_endpoint in auth_endpoints:
            try:
                auth_url = f"{base_url.rstrip('/')}{auth_endpoint}"
                
                # Try to get auth info or login
                auth_data = {
                    'username': username or 'admin',
                    'password': password or 'password'
                }
                
                auth_req = urllib.request.Request(
                    auth_url,
                    data=json.dumps(auth_data).encode(),
                    headers={'Content-Type': 'application/json'}
                )
                
                with urllib.request.urlopen(auth_req, timeout=10) as auth_response:
                    auth_result = json.loads(auth_response.read().decode())
                    
                    # Extract token if available
                    token = None
                    if 'token' in auth_result:
                        token = auth_result['token']
                    elif 'access_token' in auth_result:
                        token = auth_result['access_token']
                    elif 'jwt' in auth_result:
                        token = auth_result['jwt']
                    
                    # Now try to fetch users with token
                    if token:
                        token_headers = {
                            'Authorization': f'Bearer {token}',
                            'Content-Type': 'application/json'
                        }
                        
                        for endpoint in users_endpoints:
                            try:
                                url = f"{base_url.rstrip('/')}{endpoint}"
                                params = urllib.parse.urlencode({'page': page, 'per_page': per_page})
                                full_url = f"{url}?{params}"
                                
                                req = urllib.request.Request(full_url, headers=token_headers)
                                
                                with urllib.request.urlopen(req, timeout=10) as response:
                                    if response.status == 200:
                                        data = json.loads(response.read().decode())
                                        return {
                                            'success': True,
                                            'method': 'token_auth',
                                            'auth_endpoint': auth_endpoint,
                                            'users_endpoint': endpoint,
                                            'token': token[:20] + '...' if len(token) > 20 else token,
                                            'data': data,
                                            'page': page,
                                            'per_page': per_page
                                        }
                            except Exception:
                                continue
                        
            except Exception:
                continue
        
        return {
            'success': False,
            'error': 'Could not authenticate or fetch users',
            'attempted_auth_endpoints': auth_endpoints,
            'attempted_users_endpoints': users_endpoints,
            'base_url': base_url
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}',
            'base_url': base_url
        }