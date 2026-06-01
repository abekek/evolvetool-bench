def fetch_filtered_events(base_url, event_type=None, source=None):
    import urllib.request
    import urllib.parse
    import urllib.error
    import json
    
    try:
        # Build query parameters
        params = {}
        if event_type:
            params['type'] = event_type
        if source:
            params['source'] = source
        
        # Construct URL with query parameters
        url = base_url
        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{base_url}?{query_string}"
        
        # Make HTTP request
        request = urllib.request.Request(url)
        request.add_header('Accept', 'application/json')
        
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status == 200:
                data = response.read().decode('utf-8')
                # Validate JSON
                json.loads(data)
                return data
            else:
                return json.dumps({"error": f"HTTP {response.status}: {response.reason}"})
                
    except urllib.error.HTTPError as e:
        return json.dumps({"error": f"HTTP {e.code}: {e.reason}"})
    except urllib.error.URLError as e:
        return json.dumps({"error": f"URL Error: {str(e.reason)}"})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON response: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})