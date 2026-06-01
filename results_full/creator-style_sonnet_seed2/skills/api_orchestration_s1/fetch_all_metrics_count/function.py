def fetch_all_metrics_count(base_url="http://127.0.0.1:18080", timeout=30):
    """
    Utility: Fetches all metrics from the /api/metrics endpoint using cursor-based pagination and returns the total count.

    Args:
        base_url (str): The base URL of the API server (default: "http://127.0.0.1:18080")
        timeout (int): Request timeout in seconds (default: 30)

    Returns:
        int: Total count of all metrics fetched from the API
    """
    import urllib.request
    import urllib.parse
    import json

    total_count = 0
    cursor = None

    while True:
        # Build URL with cursor parameter if available
        url = f"{base_url}/api/metrics"
        if cursor:
            params = urllib.parse.urlencode({"cursor": cursor})
            url = f"{url}?{params}"

        try:
            # Create request with headers
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0 (compatible; API Client)')
            request.add_header('Accept', 'application/json')
            
            # Make HTTP request
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode('utf-8'))

            # Count metrics in current page
            metrics = data.get('metrics', [])
            total_count += len(metrics)

            # Check for next cursor
            cursor = data.get('next_cursor')
            if not cursor:
                break

        except urllib.error.HTTPError as e:
            if e.code == 401:
                # Try without authentication first, or handle as unauthorized
                print(f"Warning: Unauthorized access (401) - continuing without auth")
                break
            elif e.code == 404:
                print("Warning: Metrics endpoint not found (404)")
                break
            else:
                raise Exception(f"HTTP Error {e.code}: {e.reason}")
        except Exception as e:
            raise Exception(f"Failed to fetch metrics: {str(e)}")

    return total_count