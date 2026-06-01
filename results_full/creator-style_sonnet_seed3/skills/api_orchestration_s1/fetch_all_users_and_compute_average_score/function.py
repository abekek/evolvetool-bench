def fetch_all_users_and_compute_average_score(base_url="http://127.0.0.1:18080", username="", password=""):
    """
    Utility: Authenticates with a REST API, fetches all users following pagination, and computes the average score across all users.
    
    Args:
        base_url (str): The base URL of the API (default: "http://127.0.0.1:18080")
        username (str): Username for authentication (empty string for no auth)
        password (str): Password for authentication (empty string for no auth)
    
    Returns:
        float: The average score across all users, or 0.0 if no users found
    """
    import urllib.request
    import urllib.parse
    import json
    import base64
    
    # Prepare authentication header if credentials provided
    headers = {}
    if username and password:
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('ascii')
        headers['Authorization'] = f'Basic {encoded_credentials}'
    
    all_users = []
    page = 1
    
    while True:
        # Construct URL with pagination
        url = f"{base_url}/users?page={page}"
        
        # Create request
        req = urllib.request.Request(url, headers=headers)
        
        try:
            # Make request
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                # Handle different possible response structures
                if isinstance(data, list):
                    users = data
                    has_more = len(users) > 0  # Assume no more if empty list
                elif isinstance(data, dict):
                    users = data.get('users', data.get('data', []))
                    has_more = data.get('has_more', data.get('next', len(users) > 0))
                else:
                    break
                
                # Add users to collection
                all_users.extend(users)
                
                # Check if we should continue pagination
                if not has_more or len(users) == 0:
                    break
                
                page += 1
                
        except Exception as e:
            # If authentication fails or other error, try without auth on first page
            if page == 1 and username:
                req = urllib.request.Request(f"{base_url}/users?page=1")
                try:
                    with urllib.request.urlopen(req) as response:
                        data = json.loads(response.read().decode('utf-8'))
                        if isinstance(data, list):
                            all_users.extend(data)
                        elif isinstance(data, dict):
                            all_users.extend(data.get('users', data.get('data', [])))
                except:
                    pass
            break
    
    # Compute average score
    if not all_users:
        return 0.0
    
    total_score = 0.0
    valid_users = 0
    
    for user in all_users:
        if isinstance(user, dict):
            score = user.get('score', user.get('points', user.get('rating')))
            if score is not None:
                try:
                    total_score += float(score)
                    valid_users += 1
                except (ValueError, TypeError):
                    continue
    
    return total_score / valid_users if valid_users > 0 else 0.0