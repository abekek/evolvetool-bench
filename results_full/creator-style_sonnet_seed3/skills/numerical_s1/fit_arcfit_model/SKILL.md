# fit_arcfit_model

## Description

Fit an ARCFIT model specification to data using non-linear optimization.

## Usage

```python
from fit_arcfit_model import fit_arcfit_model
result = fit_arcfit_model(<spec_string>)
print(result)
```

## Inputs
spec_string (str): ARCFIT specification in format "MODEL:model_type;PARAMS:param_specs;DATA:x1,y1|x2,y2|..."
