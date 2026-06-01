def authenticate_and_fetch_user(user_id: int = 5) -> dict:
    """Authenticate with local server and fetch user details by ID."""
    import urllib.request
    import urllib.parse
    import json
    import base64
    import sys
    import traceback
    
    try:
        # Step 1: Establish HTTP session and prepare authentication credentials
        # Using basic auth with default credentials
        username = "admin"
        password = "password"
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('ascii')
        
        # Step 2: Send authentication request to http://127.0.0.1:18080 using standard auth scheme
        auth_url = "http://127.0.0.1:18080/auth"
        auth_headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'Content-Type': 'application/json'
        }
        
        auth_request = urllib.request.Request(auth_url, headers=auth_headers, method='POST')
        
        # Step 3: Verify authentication was successful and extract auth token/session data
        with urllib.request.urlopen(auth_request) as auth_response:
            if auth_response.status != 200:
                raise Exception(f"Authentication failed with status: {auth_response.status}")
            
            auth_data = json.loads(auth_response.read().decode('utf-8'))
            
            # Extract token from response (assuming token is returned in 'token' field)
            if 'token' in auth_data:
                auth_token = auth_data['token']
            else:
                # Fallback to using basic auth if no token returned
                auth_token = None
        
        # Step 4: Make authenticated GET request to fetch user with specified ID
        user_url = f"http://127.0.0.1:18080/users/{user_id}"
        
        if auth_token:
            user_headers = {
                'Authorization': f'Bearer {auth_token}',
                'Content-Type': 'application/json'
            }
        else:
            user_headers = {
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/json'
            }
        
        user_request = urllib.request.Request(user_url, headers=user_headers, method='GET')
        
        # Step 5: Parse and validate the user data response from the API
        with urllib.request.urlopen(user_request) as user_response:
            if user_response.status != 200:
                raise Exception(f"Failed to fetch user with status: {user_response.status}")
            
            user_data = json.loads(user_response.read().decode('utf-8'))
            
            # Validate that we received user data
            if not isinstance(user_data, dict):
                raise Exception("Invalid user data format received")
            
            # Ensure the user data contains expected fields
            if 'id' not in user_data:
                raise Exception("User data missing required 'id' field")
        
        # Step 6: Return the user details as a dictionary containing user information
        return user_data
        
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        return {}