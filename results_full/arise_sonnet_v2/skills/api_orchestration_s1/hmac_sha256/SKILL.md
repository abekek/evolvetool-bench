# hmac_sha256

## Description

Generate HMAC-SHA256 hash for authentication and security purposes.

## Usage

```python
from hmac_sha256 import hmac_sha256
result = hmac_sha256(<secret>, <message>)
print(result)
```

## Inputs
secret: The secret key used for HMAC generation
    message: The message to be authenticated

## Example Tests
```python
import hmac
import hashlib

def test_hmac_sha256_basic():
    """Test basic HMAC-SHA256 generation with known inputs."""
    secret = "my_secret_key"
    message = "hello world"
    
    result = hmac_sha256(secret, message)
    
    # Verify it's a valid hex string of correct length (SHA-256 = 64 hex chars)
    assert len(result) == 64
    assert all(c in '0123456789abcdef' for c in result)
    
    # Verify it matches standard library implementation
    expected = hmac.new(secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()
    assert result == expected

def test_hma
```
