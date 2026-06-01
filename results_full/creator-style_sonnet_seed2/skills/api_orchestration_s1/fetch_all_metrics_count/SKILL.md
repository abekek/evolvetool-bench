# fetch_all_metrics_count

## Description

Utility: Fetches all metrics from the /api/metrics endpoint using cursor-based pagination and returns the total count.

## Usage

```python
from fetch_all_metrics_count import fetch_all_metrics_count
result = fetch_all_metrics_count(<base_url>, <timeout>)
print(result)
```

## Inputs
base_url (str): The base URL of the API server (default: "http://127.0.0.1:18080")
    timeout (int): Request timeout in seconds (default: 30)
