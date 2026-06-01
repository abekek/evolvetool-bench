# http_get_with_headers

## Description

Make an HTTP GET request with custom headers.

## Usage

```python
from http_get_with_headers import http_get_with_headers
result = http_get_with_headers(<url>, <headers>)
print(result)
```

## Inputs
url: The URL to make the GET request to
    headers: Dictionary of custom headers to include in the request

## Example Tests
```python
import urllib.request
import urllib.error
from unittest.mock import patch, MagicMock
import json

def test_successful_request():
    """Test successful HTTP GET request with headers"""
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"status": "success"}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    
    with patch('urllib.request.urlopen', return_value=mock_response) as mock_urlopen:
        with patch('urllib.request.Request') as mock_request:
            result = http_get_with_header
```
