# fit_arcfit_power_law_model

## Description

Parse an ARCFIT model specification and fit a power law model to the provided data.

## Usage

```python
from fit_arcfit_power_law_model import fit_arcfit_power_law_model
result = fit_arcfit_power_law_model(<spec_string>)
print(result)
```

## Inputs
spec_string (str): ARCFIT specification in format 
                      "MODEL:power_law;PARAMS:a=?,b=?,c=?;DATA:x1,y1|x2,y2|..."
