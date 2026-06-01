def fetch_all_paginated_users(base_url):
    import urllib.request
    import json
    
    all_users = []
    current_cursor = None
    has_more = True
    
    while has_more:
        # Build URL with cursor parameter if we have one
        if current_cursor:
            url = f"{base_url}?cursor={current_cursor}"
        else:
            url = base_url
        
        try:
            # Make HTTP GET request
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            # Extract users from response
            if 'users' in data:
                all_users.extend(data['users'])
            elif 'data' in data:
                all_users.extend(data['data'])
            
            # Check pagination info
            pagination = data.get('pagination', {})
            has_more = pagination.get('has_more', False)
            current_cursor = pagination.get('next_cursor')
            
        except Exception as e:
            return f"Error fetching users: {str(e)}"
    
    # Extract names and count
    user_names = [user.get('name', 'Unknown') for user in all_users]
    total_count = len(all_users)
    
    # Format result
    result = f"Total users: {total_count}\n"
    result += "User names:\n"
    for name in user_names:
        result += f"- {name}\n"
    
    return result