# fetch_all_users_with_pagination

## Description

Utility: Fetches all users from a paginated API endpoint using cursor-based pagination.
Follows next_cursor values until has_more is False to retrieve complete user list.

## Usage

```python
from fetch_all_users_with_pagination import fetch_all_users_with_pagination
result = fetch_all_users_with_pagination(<base_url>)
print(result)
```

## Inputs
base_url (str): The base API endpoint URL for fetching users
