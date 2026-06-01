# decode_base64

## Description

Decode base64 encoded binary data into bytes.

## Usage

```python
from decode_base64 import decode_base64
result = decode_base64(<encoded_string>)
print(result)
```

## Inputs
encoded_string: Base64 encoded string to decode

## Returns
bytes: The decoded binary data, or empty bytes if decoding fails

## Example Tests
```python
import base64
import tempfile

def test_basic_decode():
    """Test basic base64 decoding functionality."""
    # Test with known data
    original_data = b'Hello, World!'
    encoded = base64.b64encode(original_data).decode('ascii')
    
    result = decode_base64(encoded)
    assert result == original_data

def test_binary_data_decode():
    """Test decoding of binary data."""
    # Test with binary data (bytes 0-255)
    original_data = bytes(range(256))
    encoded = base64.b64encode(original_data).decode('ascii')
    
    result = decode_base64(encoded)
    assert result == original_data

```
