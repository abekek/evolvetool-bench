# apply_bandpass_filter_to_arcsig

## Description

Apply a band-pass filter to an ARCSIG signal using FFT-based frequency domain filtering.

## Usage

```python
from apply_bandpass_filter_to_arcsig import apply_bandpass_filter_to_arcsig
result = apply_bandpass_filter_to_arcsig(<signal_string>, <filter_string>)
print(result)
```

## Inputs
signal_string (str): ARCSIG formatted signal string (e.g., "ARCSIG:v1;SR:100;LEN:128;ENC:f32le_b64;...")
    filter_string (str): Filter specification in format "BANDPASS:<low_hz>,<high_hz>"
