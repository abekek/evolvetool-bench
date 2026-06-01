# fit_arcfit_exp_decay_model

## Description

Fits an exponential decay model (y = a * exp(-b * x) + c) to data and evaluates predictions.

## Usage

```python
from fit_arcfit_exp_decay_model import fit_arcfit_exp_decay_model
result = fit_arcfit_exp_decay_model(<spec_string>)
print(result)
```

## Inputs
spec_string (str): ARCFIT specification containing model type, parameters, and data
                      Format: "MODEL:exp_decay;PARAMS:a=?,b=?,c=?;DATA:x1,y1|x2,y2|..."
