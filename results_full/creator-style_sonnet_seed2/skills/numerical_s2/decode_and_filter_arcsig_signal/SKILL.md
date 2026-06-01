# decode_and_filter_arcsig_signal

## Description

Decode ARCSIG signal format, apply band-pass filter, and compute statistics.

## Usage

```python
from decode_and_filter_arcsig_signal import decode_and_filter_arcsig_signal
result = decode_and_filter_arcsig_signal(<arcsig_string>, <filter_spec>)
print(result)
```

## Inputs
arcsig_string (str): ARCSIG formatted signal string with metadata and base64 encoded data
    filter_spec (str): Filter specification in format "BANDPASS:low_freq,high_freq"
