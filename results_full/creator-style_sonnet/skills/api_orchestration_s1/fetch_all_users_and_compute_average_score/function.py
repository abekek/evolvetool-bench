def fetch_all_users_and_compute_average_score(base_url="http://127.0.0.1:18080", username="", password=""):
    """
    Authenticate with a web service, fetch all users with pagination support, and compute average score.
    
    Utility: Connects to an HTTP API, handles authentication, follows pagination to retrieve all users,
             and calculates the average score across all retrieved users.
    
    Args:
        base_url (str): The base URL of the API service (default: "http://127.0.0.1:18080")
        username (str): Username for authentication (empty string for no auth)
        password (str): Password for authentication (empty string for no auth)
    
    Returns:
        float: The average score across all users, or 0.0 if no users found or error occurred
    """
    import urllib.request
    import urllib.parse
    import urllib.error
    import json
    import base64
    
    try:
        # Prepare authentication if credentials provided
        auth_header = {}
        if username and password:
            credentials = f"{username}:{password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            auth_header['Authorization'] = f'Basic {encoded_credentials}'
        
        all_users = []
        page = 1
        
        while True:
            # Construct URL with pagination
            url = f"{base_url}/users?page={page}"
            
            # Create request with authentication
            request = urllib.request.Request(url, headers=auth_header)
            
            # Make HTTP request
            with urllib.request.urlopen(request) as response:
                data = json.loads(response.read().decode())
            
            # Handle different possible response structures
            if isinstance(data, list):
                users = data
                has_more = len(users) > 0
            elif isinstance(data, dict):
                users = data.get('users', data.get('data', []))
                has_more = data.get('has_more', data.get('next', False))
                if has_more is False and 'total_pages' in data:
                    has_more = page < data['total_pages']
            else:
                break
            
            # Add users to collection
            all_users.extend(users)
            
            # Check if we should continue pagination
            if not has_more or len(users) == 0:
                break
                
            page += 1
        
        # Compute average score
        if not all_users:
            return 0.0
        
        total_score = 0.0
        valid_users = 0
        
        for user in all_users:
            if isinstance(user, dict) and 'score' in user:
                try:
                    score = float(user['score'])
                    total_score += score
                    valid_users += 1
                except (ValueError, TypeError):
                    continue
        
        if valid_users == 0:
            return 0.0
        
        return total_score / valid_users
        
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        return 0.0
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        return 0.0
    except json.JSONDecodeError:
        print("Error: Invalid JSON response")
        return 0.0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 0.0