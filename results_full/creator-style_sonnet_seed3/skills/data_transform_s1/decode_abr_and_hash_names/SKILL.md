# decode_abr_and_hash_names

## Description

Decode ABR (Apache Binary Records) data and compute SHA-256 hash of each record's 'name' field.

## Usage

```python
from decode_abr_and_hash_names import decode_abr_and_hash_names
result = decode_abr_and_hash_names(<base64_data>)
print(result)
```

## Inputs
base64_data (str): Base64-encoded ABR binary data containing records
