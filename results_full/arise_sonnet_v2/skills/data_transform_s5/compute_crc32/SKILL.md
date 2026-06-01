# compute_crc32

## Description

Compute CRC32 checksum for data integrity verification.

## Usage

```python
from compute_crc32 import compute_crc32
result = compute_crc32(<data>)
print(result)
```

## Inputs
data: The bytes data to compute CRC32 checksum for

## Example Tests
```python
import struct

def test_empty_data():
    """Test CRC32 of empty data"""
    result = compute_crc32(b'')
    # Empty data should have CRC32 of 0
    assert result == 0

def test_simple_data():
    """Test CRC32 of simple ASCII data"""
    data = b'hello'
    result = compute_crc32(data)
    # Should return a valid 32-bit unsigned integer
    assert isinstance(result, int)
    assert 0 <= result <= 0xFFFFFFFF

def test_different_data_different_crc():
    """Test that different data produces different CRC32"""
    data1 = b'hello'
    data2 = b'world'
    crc1 = compute_crc32(data1)
    crc2 = c
```
