# deserialize_and_filter_tpack_data

## Description

Deserialize TPACK format data from base64 string and filter records where 'available' is True.

## Usage

```python
from deserialize_and_filter_tpack_data import deserialize_and_filter_tpack_data
result = deserialize_and_filter_tpack_data(<base64_data>)
print(result)
```

## Inputs
base64_data (str): Base64 encoded TPACK binary data containing product records
