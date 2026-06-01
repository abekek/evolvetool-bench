def fetch_user_with_auth(base_url="http://127.0.0.1:18080", user_id=5, username="", password=""):
    """
    Authenticate with a server using HTTP Basic Auth and fetch a specific user's details.
    
    Utility: Connects to an HTTP server, authenticates using basic authentication,
    and retrieves user information for a given user ID.
    
    Args:
        base_url (str): The base URL of the server (default: "http://127.0.0.1:18080")
        user_id (int): The ID of the user to fetch (default: 5)
        username (str): Username for basic authentication
        password (str): Password for basic authentication
    
    Returns:
        dict: User details as returned by the API, or error information if request fails
    """
    import urllib.request
    import urllib.parse
    import urllib.error
    import json
    import base64
    
    try:
        # Construct the URL for fetching user
        url = f"{base_url.rstrip('/')}/users/{user_id}"
        
        # Create request object
        request = urllib.request.Request(url)
        
        # Add basic authentication header if credentials provided
        if username or password:
            credentials = f"{username}:{password}"
            encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('ascii')
            request.add_header('Authorization', f'Basic {encoded_credentials}')
        
        # Add content type header
        request.add_header('Content-Type', 'application/json')
        
        # Make the request
        with urllib.request.urlopen(request) as response:
            response_data = response.read().decode('utf-8')
            
            # Try to parse as JSON
            try:
                return json.loads(response_data)
            except json.JSONDecodeError:
                return {"raw_response": response_data, "status_code": response.getcode()}
                
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else ""
        return {
            "error": "HTTP Error",
            "status_code": e.code,
            "reason": e.reason,
            "body": error_body
        }
    except urllib.error.URLError as e:
        return {
            "error": "URL Error",
            "reason": str(e.reason)
        }
    except Exception as e:
        return {
            "error": "Unexpected Error",
            "message": str(e)
        }