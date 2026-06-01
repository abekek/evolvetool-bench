# generate_auth_header

## Description

Generate Authorization header for different auth schemes.
Supports Basic, Bearer, and API-Key authentication.

## Usage

```python
from generate_auth_header import generate_auth_header
result = generate_auth_header(<auth_scheme>, <username>, <password>, <token>, <api_key>)
print(result)
```
