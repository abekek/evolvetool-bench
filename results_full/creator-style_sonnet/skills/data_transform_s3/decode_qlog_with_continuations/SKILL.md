# decode_qlog_with_continuations

## Description

Decode QLOG data and merge continuation entries with their parent entries.

## Usage

```python
from decode_qlog_with_continuations import decode_qlog_with_continuations
result = decode_qlog_with_continuations(<base64_data>)
print(result)
```

## Inputs
base64_data (str): Base64-encoded QLOG binary data
