def fetch_first_page_users_with_auth(base_url, username, password):
    import urllib.request
    import urllib.error
    import base64
    
    # Generate basic auth header
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('ascii')
    auth_header = f"Basic {encoded_credentials}"
    
    # Construct URL for first page of users
    url = f"{base_url.rstrip('/')}/users?page=1"
    
    try:
        # Create request with auth header
        request = urllib.request.Request(url)
        request.add_header('Authorization', auth_header)
        
        # Make the request
        with urllib.request.urlopen(request) as response:
            return response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        return f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL Error: {e.reason}"
    except Exception as e:
        return f"Error: {str(e)}"