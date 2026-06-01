# parse_binary_data

## Description

Parse binary data using struct format specifications to extract structured fields.

## Usage

```python
from parse_binary_data import parse_binary_data
result = parse_binary_data(<data>, <format_spec>)
print(result)
```

## Inputs
data: Binary data to parse
    format_spec: Struct format string (e.g., '>I' for big-endian uint32, '<HH' for two little-endian uint16s)
                Can also be a comma-separated list of formats for sequential parsing

## Example Tests
```python
import struct
import tempfile

def test_basic_parsing():
    """Test basic struct format parsing"""
    # Create test data: big-endian uint32 (value 0x12345678)
    test_data = struct.pack('>I', 0x12345678)
    result = parse_binary_data(test_data, '>I')
    assert result == [0x12345678]

def test_multiple_values():
    """Test parsing multiple values in one format"""
    # Create test data: two little-endian uint16s
    test_data = struct.pack('<HH', 0x1234, 0x5678)
    result = parse_binary_data(test_data, '<HH')
    assert result == [0x1234, 0x5678]

def test_sequential_parsing():
    """Te
```
