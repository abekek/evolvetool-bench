# process_signal_spectrum

## Description

Decode ARCSIG signal format and compute its one-sided frequency spectrum.

## Usage

```python
from process_signal_spectrum import process_signal_spectrum
result = process_signal_spectrum(<signal_string>)
print(result)
```

## Inputs
signal_string (str): ARCSIG formatted signal string

## Example Tests
```python
import base64
import struct
import math

def test_valid_arcsig_signal():
    """Test processing a valid ARCSIG signal"""
    # Create test signal: simple sine wave
    sample_rate = 100.0
    samples = [math.sin(2 * math.pi * 5 * t / sample_rate) for t in range(32)]  # 5 Hz sine wave
    
    # Encode as ARCSIG format
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    signal_string = f'ARCSIG:{encoded_data}:{sample_rate}'
    
    result = process_signal_spectrum(signal_string)
    
    # Should return a list of [fr
```
