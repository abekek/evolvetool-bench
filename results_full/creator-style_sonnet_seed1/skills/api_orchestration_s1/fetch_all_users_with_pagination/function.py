def fetch_all_users_with_pagination(base_url="http://127.0.0.1:18080/api/users"):
    """
    Utility: Fetches all users from a paginated API endpoint using cursor-based pagination.
    Follows next_cursor values until has_more is False to retrieve complete user list.
    
    Args:
        base_url (str): The base API endpoint URL for fetching users
        
    Returns:
        dict: Contains 'total_count' (int) and 'user_names' (list of str)
    """
    import urllib.request
    import urllib.parse
    import json
    
    all_users = []
    cursor = None
    
    while True:
        # Build URL with cursor parameter if available
        if cursor:
            params = urllib.parse.urlencode({'cursor': cursor})
            url = f"{base_url}?{params}"
        else:
            url = base_url
            
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
            
            if not has_more:
                break
                
            cursor = pagination.get('next_cursor')
            if not cursor:
                break
                
        except Exception as e:
            # Return partial results if error occurs
            break
    
    # Extract user names
    user_names = []
    for user in all_users:
        if isinstance(user, dict) and 'name' in user:
            user_names.append(user['name'])
        elif isinstance(user, str):
            user_names.append(user)
    
    return {
        'total_count': len(user_names),
        'user_names': user_names
    }