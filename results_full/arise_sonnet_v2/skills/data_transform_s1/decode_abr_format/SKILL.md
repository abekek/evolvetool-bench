# decode_abr_format

## Description

Decode ARISE Binary Record (ABR) format data from base64 encoded binary.

## Usage

```python
from decode_abr_format import decode_abr_format
result = decode_abr_format(<abr_data>)
print(result)
```

## Inputs
abr_data: Base64 encoded ABR format binary data

## Example Tests
```python
import base64

def test_decode_abr_basic():
    """Test basic ABR decoding with simple key-value pairs."""
    # Create test data: record with city=NYC, temp=72
    data = bytearray()
    # First record
    data.extend([4])  # key length
    data.extend(b'city')  # key
    data.extend([3])  # value length  
    data.extend(b'NYC')  # value
    data.extend([4])  # key length
    data.extend(b'temp')  # key
    data.extend([2])  # value length
    data.extend(b'72')  # value
    data.extend([0xFF])  # record separator
    # Second record
    data.extend([4])  # key length
    data.extend(b'city'
```
