def http_get_with_headers(url: str, headers: dict) -> str:
    """
    Make an HTTP GET request with custom headers.
    
    Args:
        url: The URL to make the GET request to
        headers: Dictionary of custom headers to include in the request
    
    Returns:
        Response body as string, or error message if request fails
    """
    import urllib.request
    import urllib.error
    
    try:
        # Create request object with custom headers
        request = urllib.request.Request(url)
        
        # Add all custom headers
        for key, value in headers.items():
            request.add_header(key, str(value))
        
        # Make the request
        with urllib.request.urlopen(request) as response:
            return response.read().decode('utf-8')
            
    except urllib.error.HTTPError as e:
        return f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL Error: {e.reason}"
    except Exception as e:
        return f"Error: {str(e)}"