# authenticate_and_fetch_users

## Description

Utility: Authenticates with a web service and fetches the first page of users

## Usage

```python
from authenticate_and_fetch_users import authenticate_and_fetch_users
result = authenticate_and_fetch_users(<base_url>, <username>, <password>, <page>, <per_page>)
print(result)
```

## Inputs
base_url (str): The base URL of the API service (default: "http://127.0.0.1:18080")
    username (str): Username for authentication (default: empty string for auto-detection)
    password (str): Password for authentication (default: empty string for auto-detection)
    page (int): Page number to fetch (default: 1)
    per_page (int): Number of users per page (default: 10)
