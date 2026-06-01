# fetch_all_metrics_count

## Description

Utility: Fetches all metrics from a paginated API endpoint and returns the total count.
Uses cursor-based pagination to traverse all pages until no more data is available.

## Usage

```python
from fetch_all_metrics_count import fetch_all_metrics_count
result = fetch_all_metrics_count(<base_url>, <endpoint>)
print(result)
```

## Inputs
base_url (str): The base URL of the API server (default: "http://127.0.0.1:18080")
    endpoint (str): The API endpoint path (default: "/api/metrics")
