def fetch_all_users_with_pagination(base_url="http://127.0.0.1:18080/api/users"):
    """
    Utility: Fetches all users from a paginated API endpoint using cursor-based pagination.
    Continues fetching until has_more is False, collecting all user data across pages.
    
    Args:
        base_url (str): The base API endpoint URL for fetching users
        
    Returns:
        dict: Contains 'total_count' (int) and 'names' (list) of all users
    """
    import urllib.request
    import urllib.parse
    import json
    
    all_users = []
    cursor = None
    
    while True:
        # Build URL with cursor parameter if available
        url = base_url
        if cursor:
            params = urllib.parse.urlencode({'cursor': cursor})
            url = f"{base_url}?{params}"
        
        try:
            # Make HTTP request
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            # Extract users from response
            users = data.get('users', [])
            all_users.extend(users)
            
            # Check pagination info
            pagination = data.get('pagination', {})
            has_more = pagination.get('has_more', False)
            cursor = pagination.get('next_cursor')
            
            # Break if no more pages
            if not has_more:
                break
                
        except Exception as e:
            return {
                'error': f"Failed to fetch users: {str(e)}",
                'total_count': 0,
                'names': []
            }
    
    # Extract names from all users
    names = [user.get('name', '') for user in all_users if user.get('name')]
    
    return {
        'total_count': len(all_users),
        'names': names
    }