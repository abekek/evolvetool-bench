def authenticate_and_fetch_users(base_url="http://127.0.0.1:18080", username="", password="", page=1, per_page=10):
    """
    Utility: Authenticates with a web service and fetches the first page of users
    
    Args:
        base_url (str): The base URL of the API service (default: "http://127.0.0.1:18080")
        username (str): Username for authentication (empty string for no auth)
        password (str): Password for authentication (empty string for no auth)
        page (int): Page number to fetch (default: 1)
        per_page (int): Number of users per page (default: 10)
    
    Returns:
        dict: Response containing user data or error information with keys like 'status', 'data', 'error'
    """
    import urllib.request
    import urllib.parse
    import urllib.error
    import json
    import base64
    
    try:
        # Construct the users endpoint URL
        users_url = f"{base_url.rstrip('/')}/users"
        
        # Add pagination parameters
        params = {
            'page': str(page),
            'per_page': str(per_page)
        }
        query_string = urllib.parse.urlencode(params)
        full_url = f"{users_url}?{query_string}"
        
        # Create request
        request = urllib.request.Request(full_url)
        request.add_header('Content-Type', 'application/json')
        
        # Add authentication if credentials provided
        if username and password:
            auth_string = f"{username}:{password}"
            auth_bytes = auth_string.encode('ascii')
            auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
            request.add_header('Authorization', f'Basic {auth_b64}')
        
        # Make the request
        with urllib.request.urlopen(request, timeout=10) as response:
            response_data = response.read().decode('utf-8')
            
            # Try to parse as JSON
            try:
                parsed_data = json.loads(response_data)
                return {
                    'status': 'success',
                    'status_code': response.getcode(),
                    'data': parsed_data
                }
            except json.JSONDecodeError:
                return {
                    'status': 'success',
                    'status_code': response.getcode(),
                    'data': response_data
                }
                
    except urllib.error.HTTPError as e:
        try:
            error_data = e.read().decode('utf-8')
            try:
                error_json = json.loads(error_data)
                return {
                    'status': 'error',
                    'status_code': e.code,
                    'error': error_json
                }
            except json.JSONDecodeError:
                return {
                    'status': 'error',
                    'status_code': e.code,
                    'error': error_data
                }
        except:
            return {
                'status': 'error',
                'status_code': e.code,
                'error': str(e)
            }
            
    except urllib.error.URLError as e:
        return {
            'status': 'error',
            'error': f"Connection error: {str(e)}"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f"Unexpected error: {str(e)}"
        }