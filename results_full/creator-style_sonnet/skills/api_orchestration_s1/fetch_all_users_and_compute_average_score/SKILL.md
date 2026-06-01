# fetch_all_users_and_compute_average_score

## Description

Authenticate with a web service, fetch all users with pagination support, and compute average score.

## Usage

```python
from fetch_all_users_and_compute_average_score import fetch_all_users_and_compute_average_score
result = fetch_all_users_and_compute_average_score(<base_url>, <username>, <password>)
print(result)
```

## Inputs
base_url (str): The base URL of the API service (default: "http://127.0.0.1:18080")
    username (str): Username for authentication (empty string for no auth)
    password (str): Password for authentication (empty string for no auth)
