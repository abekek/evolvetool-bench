def make_http_get_request(url):
    """
    Makes a GET request to the specified URL and returns detailed response information or error details.
    
    Utility: Performs HTTP GET requests and captures both successful responses and error conditions,
             providing detailed information about what occurred during the request.
    
    Args:
        url (str): The URL to make the GET request to
    
    Returns:
        dict: Contains 'status', 'error' (if any), 'response_data', 'headers', and 'status_code'
    """
    import urllib.request
    import urllib.error
    import json
    
    result = {
        'status': 'success',
        'error': None,
        'response_data': None,
        'headers': {},
        'status_code': None
    }
    
    try:
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Python-urllib/3.x')
        
        with urllib.request.urlopen(request) as response:
            result['status_code'] = response.getcode()
            result['headers'] = dict(response.headers)
            
            # Read response data
            response_data = response.read().decode('utf-8')
            
            # Try to parse as JSON, fallback to raw text
            try:
                result['response_data'] = json.loads(response_data)
            except json.JSONDecodeError:
                result['response_data'] = response_data
                
    except urllib.error.HTTPError as e:
        result['status'] = 'http_error'
        result['status_code'] = e.code
        result['error'] = f"HTTP {e.code}: {e.reason}"
        result['headers'] = dict(e.headers) if e.headers else {}
        
        # Try to read error response body
        try:
            error_body = e.read().decode('utf-8')
            try:
                result['response_data'] = json.loads(error_body)
            except json.JSONDecodeError:
                result['response_data'] = error_body
        except:
            result['response_data'] = None
            
    except urllib.error.URLError as e:
        result['status'] = 'connection_error'
        result['error'] = f"Connection error: {str(e.reason)}"
        
    except Exception as e:
        result['status'] = 'unexpected_error'
        result['error'] = f"Unexpected error: {str(e)}"
    
    return result