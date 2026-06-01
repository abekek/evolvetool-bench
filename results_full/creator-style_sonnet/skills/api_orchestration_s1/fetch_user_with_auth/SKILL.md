# fetch_user_with_auth

## Description

Authenticate with a server using HTTP Basic Auth and fetch a specific user's details.

## Usage

```python
from fetch_user_with_auth import fetch_user_with_auth
result = fetch_user_with_auth(<base_url>, <user_id>, <username>, <password>)
print(result)
```

## Inputs
base_url (str): The base URL of the server (default: "http://127.0.0.1:18080")
    user_id (int): The ID of the user to fetch (default: 5)
    username (str): Username for basic authentication
    password (str): Password for basic authentication
