def fetch_authenticated_user(base_url="http://127.0.0.1:18080", user_id=5, username="admin", password="password"):
    """
    Authenticate with a web service and fetch user details by ID.
    
    Utility: Performs HTTP basic authentication and retrieves user information from a REST API endpoint.
    
    Args:
        base_url (str): The base URL of the web service (default: "http://127.0.0.1:18080")
        user_id (int): The ID of the user to fetch (default: 5)
        username (str): Username for authentication (default: "admin")
        password (str): Password for authentication (default: "password")
    
    Returns:
        dict: User details as a dictionary, or error information if the request fails
    """
    import urllib.request
    import urllib.error
    import json
    import base64
    
    # Create the full URL for the user endpoint
    url = f"{base_url}/users/{user_id}"
    
    # Create basic auth header
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('ascii')
    auth_header = f"Basic {encoded_credentials}"
    
    # Create request with authentication header
    request = urllib.request.Request(url)
    request.add_header("Authorization", auth_header)
    request.add_header("Content-Type", "application/json")
    
    try:
        # Make the request
        with urllib.request.urlopen(request) as response:
            response_data = response.read().decode('utf-8')
            return json.loads(response_data)
    
    except urllib.error.HTTPError as e:
        return {
            "error": f"HTTP {e.code}",
            "message": e.reason,
            "url": url
        }
    except urllib.error.URLError as e:
        return {
            "error": "Connection failed",
            "message": str(e.reason),
            "url": url
        }
    except json.JSONDecodeError as e:
        return {
            "error": "Invalid JSON response",
            "message": str(e),
            "url": url
        }
    except Exception as e:
        return {
            "error": "Unexpected error",
            "message": str(e),
            "url": url
        }