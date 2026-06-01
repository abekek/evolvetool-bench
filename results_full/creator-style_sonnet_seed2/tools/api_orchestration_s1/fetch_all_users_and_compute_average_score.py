def fetch_all_users_and_compute_average_score(base_url="http://127.0.0.1:18080", username="", password=""):
    """
    Authenticates with a web service, fetches all users through pagination, and computes average score.
    
    Utility: Connects to an API endpoint, handles authentication and pagination to retrieve all user data,
    then calculates the average score across all users.
    
    Args:
        base_url (str): The base URL of the API service (default: "http://127.0.0.1:18080")
        username (str): Username for authentication (empty string for no auth)
        password (str): Password for authentication (empty string for no auth)
    
    Returns:
        float: The average score across all users, or 0.0 if no users found or error occurred
    """
    import urllib.request
    import urllib.parse
    import json
    import base64
    
    try:
        all_users = []
        page = 1
        
        while True:
            # Construct URL for current page
            url = f"{base_url}/users?page={page}"
            
            # Create request
            request = urllib.request.Request(url)
            
            # Add authentication if credentials provided
            if username and password:
                credentials = f"{username}:{password}"
                encoded_credentials = base64.b64encode(credentials.encode()).decode()
                request.add_header("Authorization", f"Basic {encoded_credentials}")
            
            # Add headers
            request.add_header("Content-Type", "application/json")
            request.add_header("Accept", "application/json")
            
            try:
                # Make request
                with urllib.request.urlopen(request) as response:
                    data = json.loads(response.read().decode())
                    
                    # Handle different possible response structures
                    users = []
                    if isinstance(data, list):
                        users = data
                    elif isinstance(data, dict):
                        if 'users' in data:
                            users = data['users']
                        elif 'data' in data:
                            users = data['data']
                        elif 'results' in data:
                            users = data['results']
                    
                    # If no users found on this page, we've reached the end
                    if not users:
                        break
                    
                    all_users.extend(users)
                    
                    # Check if there are more pages
                    has_more = False
                    if isinstance(data, dict):
                        has_more = data.get('has_more', False) or data.get('hasMore', False)
                        total_pages = data.get('total_pages', data.get('totalPages', 0))
                        if total_pages > 0 and page >= total_pages:
                            break
                    
                    # If we got fewer users than expected page size, probably last page
                    if len(users) < 20:  # Assume common page size of 20
                        break
                    
                    if not has_more and isinstance(data, dict):
                        break
                    
                    page += 1
                    
            except urllib.error.HTTPError as e:
                if e.code == 404 and page > 1:
                    # Reached end of pagination
                    break
                else:
                    return 0.0
            except Exception:
                return 0.0
        
        # Calculate average score
        if not all_users:
            return 0.0
        
        total_score = 0.0
        valid_users = 0
        
        for user in all_users:
            if isinstance(user, dict):
                # Try different possible field names for score
                score = user.get('score') or user.get('Score') or user.get('rating') or user.get('points')
                if score is not None:
                    try:
                        total_score += float(score)
                        valid_users += 1
                    except (ValueError, TypeError):
                        continue
        
        if valid_users == 0:
            return 0.0
        
        return round(total_score / valid_users, 2)
        
    except Exception:
        return 0.0