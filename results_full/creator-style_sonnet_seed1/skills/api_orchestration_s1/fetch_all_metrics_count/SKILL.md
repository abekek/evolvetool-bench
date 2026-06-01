# fetch_all_metrics_count

## Description

Fetches all metrics from the /api/metrics endpoint using cursor-based pagination
and returns the total count of metrics retrieved.

## Usage

```python
from fetch_all_metrics_count import fetch_all_metrics_count
result = fetch_all_metrics_count(<base_url>)
print(result)
```

## Inputs
base_url (str): The base URL of the API server
