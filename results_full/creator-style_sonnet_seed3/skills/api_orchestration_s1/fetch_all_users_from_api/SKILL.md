# fetch_all_users_from_api

## Description

Utility: Fetches all users from a paginated API endpoint using cursor-based pagination.
Follows pagination cursors until all users are retrieved and returns summary statistics.

## Usage

```python
from fetch_all_users_from_api import fetch_all_users_from_api
result = fetch_all_users_from_api(<base_url>)
print(result)
```

## Inputs
base_url (str): The base URL for the users API endpoint. Defaults to "http://127.0.0.1:18080/api/users"
