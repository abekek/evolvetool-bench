# parse_and_fit_arcfit_model

## Description

Parse and fit an ARCFIT model specification to data using non-linear least squares.

## Usage

```python
from parse_and_fit_arcfit_model import parse_and_fit_arcfit_model
result = parse_and_fit_arcfit_model(<arcfit_spec>)
print(result)
```

## Inputs
arcfit_spec (str): ARCFIT specification string in format 
                      "MODEL:<name>;PARAMS:<key>=<val_or_?>,...;DATA:<x1>,<y1>|<x2>,<y2>|..."
