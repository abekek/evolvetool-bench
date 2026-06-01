# apply_bandpass_filter_to_arcsig

## Description

Apply a band-pass filter to an ARCSIG signal using FFT-based filtering.

## Usage

```python
from apply_bandpass_filter_to_arcsig import apply_bandpass_filter_to_arcsig
result = apply_bandpass_filter_to_arcsig(<arcsig_signal>, <filter_spec>)
print(result)
```

## Inputs
arcsig_signal (str): ARCSIG formatted signal string
    filter_spec (str): Filter specification in format "BANDPASS:<low_hz>,<high_hz>"
