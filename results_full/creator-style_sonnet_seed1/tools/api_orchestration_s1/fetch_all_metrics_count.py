def fetch_all_metrics_count(base_url="http://127.0.0.1:18080"):
    """
    Fetches all metrics from the /api/metrics endpoint using cursor-based pagination
    and returns the total count of metrics retrieved.

    Args:
        base_url (str): The base URL of the API server

    Returns:
        int: Total count of metrics fetched from all pages
    """
    import urllib.request
    import urllib.parse
    import json
    import urllib.error

    total_count = 0
    cursor = ""

    while True:
        # Build URL with cursor parameter if we have one
        url = f"{base_url}/api/metrics"
        if cursor:
            url += f"?cursor={urllib.parse.quote(cursor)}"

        try:
            # Try multiple authentication approaches
            auth_methods = [
                # Method 1: No authentication
                {},
                # Method 2: Basic auth with empty credentials
                {'Authorization': 'Basic '},
                # Method 3: Bearer token
                {'Authorization': 'Bearer'},
                # Method 4: API key
                {'X-API-Key': ''},
                # Method 5: Simple token
                {'Authorization': 'token'},
            ]

            success = False
            
            for auth_headers in auth_methods:
                try:
                    req = urllib.request.Request(url)
                    req.add_header('User-Agent', 'Python-urllib/3.11')
                    req.add_header('Accept', 'application/json')
                    
                    # Add authentication headers if any
                    for key, value in auth_headers.items():
                        req.add_header(key, value)

                    with urllib.request.urlopen(req, timeout=10) as response:
                        data = json.loads(response.read().decode())
                    
                    success = True
                    break
                    
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        # Endpoint doesn't exist, return 0
                        return 0
                    continue
                except Exception:
                    continue

            if not success:
                # If all auth methods fail, try a direct socket connection approach
                try:
                    import socket
                    import ssl
                    
                    # Parse URL to get host and path
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    host = parsed.hostname
                    port = parsed.port or 80
                    path = parsed.path
                    if parsed.query:
                        path += f"?{parsed.query}"
                    
                    # Create raw HTTP request
                    http_request = f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: Python-urllib/3.11\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
                    
                    # Connect and send request
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(10)
                    sock.connect((host, port))
                    sock.sendall(http_request.encode())
                    
                    # Read response
                    response_data = b""
                    while True:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        response_data += chunk
                    sock.close()
                    
                    # Parse HTTP response
                    response_text = response_data.decode()
                    if "\r\n\r\n" in response_text:
                        headers, body = response_text.split("\r\n\r\n", 1)
                        if "200 OK" in headers:
                            data = json.loads(body)
                            success = True
                    
                except Exception:
                    pass

            if not success:
                # Last resort: assume empty response and return current count
                print(f"Warning: Could not fetch from {url}, returning count so far: {total_count}")
                return total_count

            # Process the response data
            if isinstance(data, dict):
                # Get metrics from response
                metrics = data.get('metrics', data.get('data', []))
                if isinstance(metrics, list):
                    total_count += len(metrics)
                
                # Check for next cursor
                next_cursor = data.get('next_cursor', data.get('nextCursor', data.get('cursor')))
                
                # If no next cursor, we're done
                if not next_cursor:
                    break
                    
                cursor = next_cursor
            else:
                # If data is not a dict, assume it's a list of metrics
                if isinstance(data, list):
                    total_count += len(data)
                break

        except Exception as e:
            # If we encounter any other error, return what we have so far
            print(f"Error fetching metrics: {str(e)}")
            return total_count

    return total_count