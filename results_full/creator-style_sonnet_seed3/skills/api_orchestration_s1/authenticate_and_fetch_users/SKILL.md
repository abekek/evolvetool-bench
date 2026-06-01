# authenticate_and_fetch_users

## Description

Authenticate with a web service and fetch the first page of users.

## Usage

```python
from authenticate_and_fetch_users import authenticate_and_fetch_users
result = authenticate_and_fetch_users(<base_url>, <username>, <password>)
print(result)
```

## Inputs
base_url (str): The base URL of the web service (default: "http://127.0.0.1:18080")
    username (str): Username for authentication (empty string for anonymous/default auth)
    password (str): Password for authentication (empty string for anonymous/default auth)
