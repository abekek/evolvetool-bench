def fetch_user_with_basic_auth(base_url="http://127.0.0.1:18080", user_id=5, username="", password=""):
    """
    Utility: Authenticates with a web service using HTTP Basic Authentication and fetches user details by ID
    
    Args:
        base_url (str): The base URL of the API service (default: http://127.0.0.1:18080)
        user_id (int): The ID of the user to fetch (default: 5)
        username (str): Username for basic authentication
        password (str): Password for basic authentication
    
    Returns:
        dict: User details from the API response, or error information if request fails
    """
    import urllib.request
    import urllib.error
    import json
    import base64
    
    try:
        # Construct the URL for fetching user by ID
        url = f"{base_url.rstrip('/')}/user/{user_id}"
        
        # Create request object
        request = urllib.request.Request(url)
        
        # Add Basic Authentication header if credentials provided
        if username and password:
            credentials = f"{username}:{password}"
            encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('ascii')
            request.add_header('Authorization', f'Basic {encoded_credentials}')
        
        # Add headers for JSON response
        request.add_header('Accept', 'application/json')
        request.add_header('Content-Type', 'application/json')
        
        # Make the request
        with urllib.request.urlopen(request, timeout=10) as response:
            response_data = response.read().decode('utf-8')
            
            # Try to parse as JSON
            try:
                user_data = json.loads(response_data)
                return user_data
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response", "raw_response": response_data}
                
    except urllib.error.HTTPError as e:
        return {
            "error": f"HTTP {e.code}: {e.reason}",
            "url": url,
            "status_code": e.code
        }
    except urllib.error.URLError as e:
        return {
            "error": f"URL Error: {e.reason}",
            "url": url
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "url": url
        }