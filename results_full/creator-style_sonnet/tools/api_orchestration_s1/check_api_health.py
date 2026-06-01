def check_api_health(host="127.0.0.1", port=18080, endpoint="/health", timeout=5):
    """
    Check if an API server is healthy by making a GET request to the health endpoint.
    
    Utility: Makes an HTTP GET request to check API server health status and returns detailed response information
    
    Args:
        host (str): The server hostname or IP address (default: "127.0.0.1")
        port (int): The server port number (default: 18080)
        endpoint (str): The health check endpoint path (default: "/health")
        timeout (int): Request timeout in seconds (default: 5)
    
    Returns:
        dict: Contains 'status' (healthy/unhealthy/error), 'status_code', 'response_time_ms', 'message', and 'response_body'
    """
    import urllib.request
    import urllib.error
    import time
    import json
    
    url = f"http://{host}:{port}{endpoint}"
    start_time = time.time()
    
    try:
        request = urllib.request.Request(url, method='GET')
        request.add_header('User-Agent', 'Python-Health-Checker/1.0')
        
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_time_ms = round((time.time() - start_time) * 1000, 2)
            status_code = response.getcode()
            response_body = response.read().decode('utf-8')
            
            if status_code == 200:
                return {
                    'status': 'healthy',
                    'status_code': status_code,
                    'response_time_ms': response_time_ms,
                    'message': 'API server is healthy',
                    'response_body': response_body
                }
            else:
                return {
                    'status': 'unhealthy',
                    'status_code': status_code,
                    'response_time_ms': response_time_ms,
                    'message': f'API returned non-200 status code: {status_code}',
                    'response_body': response_body
                }
                
    except urllib.error.HTTPError as e:
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        return {
            'status': 'unhealthy',
            'status_code': e.code,
            'response_time_ms': response_time_ms,
            'message': f'HTTP Error: {e.reason}',
            'response_body': str(e)
        }
        
    except urllib.error.URLError as e:
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        return {
            'status': 'error',
            'status_code': None,
            'response_time_ms': response_time_ms,
            'message': f'Connection Error: {e.reason}',
            'response_body': str(e)
        }
        
    except Exception as e:
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        return {
            'status': 'error',
            'status_code': None,
            'response_time_ms': response_time_ms,
            'message': f'Unexpected error: {str(e)}',
            'response_body': str(e)
        }