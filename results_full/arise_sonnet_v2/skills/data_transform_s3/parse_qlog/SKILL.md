# parse_qlog

## Description

Parse QLOG binary format data into structured records.

## Usage

```python
from parse_qlog import parse_qlog
result = parse_qlog(<binary_data>)
print(result)
```

## Inputs
binary_data: Raw binary data in QLOG format

## Example Tests
```python
import struct
import datetime

def test_parse_single_qlog_record():
    """Test parsing a single QLOG record."""
    # Create test data
    timestamp = 1640995200000000  # 2022-01-01 00:00:00 in microseconds
    flags = 0x01
    message = b"Test message"
    length = len(message)
    
    # Pack into QLOG format
    binary_data = struct.pack('<QBH', timestamp, flags, length) + message
    
    result = parse_qlog(binary_data)
    
    assert len(result) == 1
    record = result[0]
    assert 'error' not in record
    assert record['timestamp'] == timestamp
    assert record['flags'] == flags
 
```
