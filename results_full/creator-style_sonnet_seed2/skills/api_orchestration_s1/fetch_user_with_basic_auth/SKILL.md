# fetch_user_with_basic_auth

## Description

Utility: Authenticates with a web service using HTTP Basic Authentication and fetches user details by ID

## Usage

```python
from fetch_user_with_basic_auth import fetch_user_with_basic_auth
result = fetch_user_with_basic_auth(<base_url>, <user_id>, <username>, <password>)
print(result)
```

## Inputs
base_url (str): The base URL of the API service (default: http://127.0.0.1:18080)
    user_id (int): The ID of the user to fetch (default: 5)
    username (str): Username for basic authentication
    password (str): Password for basic authentication
