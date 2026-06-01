def fetch_all_paginated_users(base_url, endpoint_path):
    import urllib.request
    import json
    
    all_users = []
    cursor = None
    has_more = True
    
    while has_more:
        # Construct URL with cursor parameter if available
        if cursor:
            url = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}?cursor={cursor}"
        else:
            url = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"
        
        try:
            # Make HTTP request
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            # Extract users from response
            if 'users' in data:
                all_users.extend(data['users'])
            
            # Check pagination info
            pagination = data.get('pagination', {})
            has_more = pagination.get('has_more', False)
            cursor = pagination.get('next_cursor')
            
        except Exception as e:
            return f"Error fetching users: {str(e)}"
    
    # Extract user names
    user_names = []
    for user in all_users:
        if isinstance(user, dict) and 'name' in user:
            user_names.append(user['name'])
        elif isinstance(user, str):
            user_names.append(user)
    
    # Format result
    total_count = len(all_users)
    names_str = ', '.join(user_names) if user_names else 'No names available'
    
    return f"Total users: {total_count}\nUser names: {names_str}"