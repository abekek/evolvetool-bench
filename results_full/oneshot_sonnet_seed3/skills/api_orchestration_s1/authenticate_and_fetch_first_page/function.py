def authenticate_and_fetch_first_page(base_url, username, password):
    import urllib.request
    import urllib.error
    import base64
    import json
    
    try:
        # Create basic auth header
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('ascii')
        auth_header = f"Basic {encoded_credentials}"
        
        # Construct users endpoint URL
        if base_url.endswith('/'):
            users_url = f"{base_url}users"
        else:
            users_url = f"{base_url}/users"
        
        # Make authenticated request
        request = urllib.request.Request(users_url)
        request.add_header('Authorization', auth_header)
        request.add_header('Accept', 'application/json')
        
        with urllib.request.urlopen(request) as response:
            content = response.read().decode('utf-8')
            return content
            
    except urllib.error.HTTPError as e:
        return f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL Error: {e.reason}"
    except Exception as e:
        return f"Error: {str(e)}"