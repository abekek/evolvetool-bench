# deserialize_tpack

## Description

Deserialize TPACK (binary packed) data format into structured records.

## Usage

```python
from deserialize_tpack import deserialize_tpack
result = deserialize_tpack(<data>)
print(result)
```

## Inputs
data (str): Base64-encoded TPACK binary data

## Example Tests
```python
import base64
import struct

def create_tpack_data(schema, records):
    """Helper to create valid TPACK data for testing"""
    data = bytearray()
    
    # Header: magic + record count
    data.extend(struct.pack('<II', 0x54504143, len(records)))
    
    # Schema: field count + fields
    data.extend(struct.pack('<I', len(schema)))
    for field_type, field_name in schema:
        data.append(field_type)
        data.extend(field_name.encode('utf-8'))
        data.append(0)  # null terminator
    
    # Records
    for record in records:
        for field_type, field_name in schema:
      
```
