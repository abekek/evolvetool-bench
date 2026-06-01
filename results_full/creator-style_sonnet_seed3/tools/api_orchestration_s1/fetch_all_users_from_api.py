def fetch_all_users_from_api(base_url="http://127.0.0.1:18080/api/users"):
    """
    Utility: Fetches all users from a paginated API endpoint using cursor-based pagination.
    Follows pagination cursors until all users are retrieved and returns summary statistics.
    
    Args:
        base_url (str): The base URL for the users API endpoint. Defaults to "http://127.0.0.1:18080/api/users"
    
    Returns:
        dict: A dictionary containing:
            - 'total_users' (int): Total number of users fetched
            - 'user_names' (list): List of all user names
            - 'status' (str): Status message indicating success or failure
    """
    import urllib.request
    import urllib.parse
    import json
    
    all_users = []
    next_cursor = None
    
    try:
        while True:
            # Build URL with cursor parameter if available
            if next_cursor:
                url = f"{base_url}?cursor={urllib.parse.quote(next_cursor)}"
            else:
                url = base_url
            
            # Make HTTP request
            with urllib.request.urlopen(url) as response:
                if response.status != 200:
                    return {
                        'total_users': 0,
                        'user_names': [],
                        'status': f'Error: HTTP {response.status}'
                    }
                
                data = json.loads(response.read().decode('utf-8'))
            
            # Extract users from response
            if 'users' in data:
                all_users.extend(data['users'])
            
            # Check pagination
            pagination = data.get('pagination', {})
            has_more = pagination.get('has_more', False)
            next_cursor = pagination.get('next_cursor')
            
            # Break if no more pages
            if not has_more:
                break
        
        # Extract user names
        user_names = []
        for user in all_users:
            if isinstance(user, dict) and 'name' in user:
                user_names.append(user['name'])
            elif isinstance(user, str):
                user_names.append(user)
        
        return {
            'total_users': len(all_users),
            'user_names': user_names,
            'status': 'Success'
        }
        
    except Exception as e:
        return {
            'total_users': 0,
            'user_names': [],
            'status': f'Error: {str(e)}'
        }