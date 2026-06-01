# check_api_health

## Description

Check if an API server is healthy by making a GET request to the health endpoint.

## Usage

```python
from check_api_health import check_api_health
result = check_api_health(<host>, <port>, <endpoint>, <timeout>)
print(result)
```

## Inputs
host (str): The server hostname or IP address (default: "127.0.0.1")
    port (int): The server port number (default: 18080)
    endpoint (str): The health check endpoint path (default: "/health")
    timeout (int): Request timeout in seconds (default: 5)
