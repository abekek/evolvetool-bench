# parse_and_fit_arcfit_model

## Description

Parse and fit an ARCFIT model specification using non-linear least squares.

## Usage

```python
from parse_and_fit_arcfit_model import parse_and_fit_arcfit_model
result = parse_and_fit_arcfit_model(<spec_string>)
print(result)
```

## Inputs
spec_string (str): ARCFIT specification in format 
                      MODEL:<name>;PARAMS:<key>=<val_or_?>,...;DATA:<x1>,<y1>|<x2>,<y2>|...
