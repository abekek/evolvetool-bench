def authenticate_and_fetch_users(base_url="http://127.0.0.1:18080", username="", password="", page=1, per_page=10):
    """
    Authenticate with a web service and fetch the first page of users.
    
    Utility: Performs HTTP authentication (tries both basic auth and form-based auth) 
    and retrieves user data from the /users endpoint. Handles common authentication 
    methods and returns user data in a structured format.
    
    Args:
        base_url (str): The base URL of the API service (default: "http://127.0.0.1:18080")
        username (str): Username for authentication (default: empty string for anonymous)
        password (str): Password for authentication (default: empty string)
        page (int): Page number to fetch (default: 1)
        per_page (int): Number of users per page (default: 10)
    
    Returns:
        dict: Contains 'success' (bool), 'data' (list of users or error message), 
              'status_code' (int), and 'auth_method' (str) used for successful auth
    """
    import urllib.request
    import urllib.parse
    import urllib.error
    import json
    import base64
    
    def try_request(url, headers=None, data=None, method='GET'):
        if headers is None:
            headers = {}
        
        if data and isinstance(data, dict):
            data = urllib.parse.urlencode(data).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
                try:
                    return {'success': True, 'data': json.loads(content), 'status_code': response.status}
                except json.JSONDecodeError:
                    return {'success': True, 'data': content, 'status_code': response.status}
        except urllib.error.HTTPError as e:
            return {'success': False, 'data': f'HTTP Error {e.code}: {e.reason}', 'status_code': e.code}
        except Exception as e:
            return {'success': False, 'data': f'Request failed: {str(e)}', 'status_code': 0}
    
    # Prepare users endpoint URL
    users_url = f"{base_url.rstrip('/')}/users"
    if page > 1 or per_page != 10:
        params = urllib.parse.urlencode({'page': page, 'per_page': per_page})
        users_url += f"?{params}"
    
    # Method 1: Try without authentication first
    result = try_request(users_url)
    if result['success'] and result['status_code'] == 200:
        return {
            'success': True,
            'data': result['data'],
            'status_code': result['status_code'],
            'auth_method': 'none'
        }
    
    # If authentication is needed and credentials provided
    if username or password:
        # Method 2: Try Basic Authentication
        if username and password:
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers = {'Authorization': f'Basic {credentials}'}
            result = try_request(users_url, headers=headers)
            if result['success'] and result['status_code'] == 200:
                return {
                    'success': True,
                    'data': result['data'],
                    'status_code': result['status_code'],
                    'auth_method': 'basic'
                }
        
        # Method 3: Try form-based login first, then fetch users
        login_url = f"{base_url.rstrip('/')}/login"
        login_data = {'username': username, 'password': password}
        login_result = try_request(login_url, data=login_data, method='POST')
        
        if login_result['success'] and login_result['status_code'] in [200, 302]:
            # Try to extract session token or cookie info from login response
            headers = {}
            if isinstance(login_result['data'], dict) and 'token' in login_result['data']:
                headers['Authorization'] = f"Bearer {login_result['data']['token']}"
            
            result = try_request(users_url, headers=headers)
            if result['success'] and result['status_code'] == 200:
                return {
                    'success': True,
                    'data': result['data'],
                    'status_code': result['status_code'],
                    'auth_method': 'form_login'
                }
    
    # Return the last failed attempt
    return {
        'success': False,
        'data': result['data'] if 'result' in locals() else 'Authentication failed - no valid credentials provided',
        'status_code': result.get('status_code', 401) if 'result' in locals() else 401,
        'auth_method': 'failed'
    }