# fetch_authenticated_user

## Description

Authenticate with a web service and fetch user details by ID.

## Usage

```python
from fetch_authenticated_user import fetch_authenticated_user
result = fetch_authenticated_user(<base_url>, <user_id>, <username>, <password>)
print(result)
```

## Inputs
base_url (str): The base URL of the web service (default: "http://127.0.0.1:18080")
    user_id (int): The ID of the user to fetch (default: 5)
    username (str): Username for authentication (default: "admin")
    password (str): Password for authentication (default: "password")
