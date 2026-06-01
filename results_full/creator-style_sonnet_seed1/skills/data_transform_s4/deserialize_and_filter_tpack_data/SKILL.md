# deserialize_and_filter_tpack_data

## Description

Deserializes TPACK (msgpack) encoded data and filters records where 'available' is True.

## Usage

```python
from deserialize_and_filter_tpack_data import deserialize_and_filter_tpack_data
result = deserialize_and_filter_tpack_data(<encoded_data>)
print(result)
```

## Inputs
encoded_data (str): Base64-encoded msgpack data containing product records
