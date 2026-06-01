def fetch_all_users_with_pagination(base_url="http://127.0.0.1:18080/api/users"):
    """
    Fetch all users from a paginated API endpoint using cursor-based pagination.
    
    Utility: Retrieves all users by following pagination cursors until no more data exists.
             Handles cursor-based pagination automatically and aggregates results.
    
    Args:
        base_url (str): The base API endpoint URL for fetching users
    
    Returns:
        dict: Contains 'total_count' (int) and 'names' (list of str) of all users
    """
    import urllib.request
    import urllib.parse
    import json
    
    all_users = []
    next_cursor = None
    has_more = True
    
    while has_more:
        # Build URL with cursor parameter if available
        url = base_url
        if next_cursor:
            params = urllib.parse.urlencode({'cursor': next_cursor})
            url = f"{base_url}?{params}"
        
        try:
            # Make HTTP request
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            # Extract users from current page
            if 'users' in data:
                all_users.extend(data['users'])
            
            # Update pagination info
            pagination = data.get('pagination', {})
            has_more = pagination.get('has_more', False)
            next_cursor = pagination.get('next_cursor')
            
        except Exception as e:
            return {
                'total_count': 0,
                'names': [],
                'error': f"Failed to fetch users: {str(e)}"
            }
    
    # Extract names from users
    user_names = []
    for user in all_users:
        if isinstance(user, dict) and 'name' in user:
            user_names.append(user['name'])
        elif isinstance(user, str):
            user_names.append(user)
    
    return {
        'total_count': len(all_users),
        'names': user_names
    }