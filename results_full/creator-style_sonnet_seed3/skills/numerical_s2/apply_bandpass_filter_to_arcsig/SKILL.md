# apply_bandpass_filter_to_arcsig

## Description

Apply a band-pass filter to an ARCSIG signal using FFT.

## Usage

```python
from apply_bandpass_filter_to_arcsig import apply_bandpass_filter_to_arcsig
result = apply_bandpass_filter_to_arcsig(<arcsig_string>, <filter_spec>)
print(result)
```

## Inputs
arcsig_string (str): ARCSIG format string (e.g., "ARCSIG:v1;SR:100;LEN:128;ENC:f32le_b64;...")
    filter_spec (str): Filter specification in format "BANDPASS:<low_hz>,<high_hz>"
