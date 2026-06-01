# decode_base64

## Description

Decode base64 encoded data into binary format.

## Usage

```python
from decode_base64 import decode_base64
result = decode_base64(<encoded_data>)
print(result)
```

## Inputs
encoded_data: Base64 encoded string to decode

## Example Tests
```python
import base64

def test_decode_valid_base64():
    """Test decoding valid base64 data."""
    # Test with known text
    original_text = "Hello, World!"
    encoded = base64.b64encode(original_text.encode('utf-8')).decode('ascii')
    
    result = decode_base64(encoded)
    assert isinstance(result, bytes)
    assert result == original_text.encode('utf-8')
    assert result.decode('utf-8') == original_text

def test_decode_binary_data():
    """Test decoding base64 encoded binary data."""
    # Create some binary data
    original_binary = bytes([0, 1, 2, 3, 255, 254, 253])
    encoded = base
```
