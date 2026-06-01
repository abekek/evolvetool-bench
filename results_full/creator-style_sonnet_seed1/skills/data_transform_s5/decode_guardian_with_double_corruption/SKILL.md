# decode_guardian_with_double_corruption

## Description

Decode GUARDIAN data format and attempt to repair corrupted blocks using XOR parity.
Detects when multiple blocks in the same parity group are corrupted and reports repair failure.

## Usage

```python
from decode_guardian_with_double_corruption import decode_guardian_with_double_corruption
result = decode_guardian_with_double_corruption(<encoded_data>)
print(result)
```

## Inputs
encoded_data (str): Base64 encoded GUARDIAN data with potential corruptions
