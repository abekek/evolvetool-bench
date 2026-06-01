def fetch_all_paginated_metrics(base_url, endpoint_path):
    import urllib.request
    import json
    
    total_count = 0
    cursor = None
    
    while True:
        # Build URL with cursor parameter if we have one
        if cursor:
            url = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}?cursor={cursor}"
        else:
            url = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"
        
        try:
            # Make HTTP request
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            # Count metrics in current page
            if 'data' in data and isinstance(data['data'], list):
                total_count += len(data['data'])
            
            # Check for next cursor
            if 'next_cursor' in data and data['next_cursor']:
                cursor = data['next_cursor']
            else:
                break
                
        except Exception as e:
            return f"Error fetching metrics: {str(e)}"
    
    return str(total_count)