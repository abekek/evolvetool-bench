# fit_arcfit_exponential_decay

## Description

Fits an exponential decay model to data and evaluates predictions with statistics.

## Usage

```python
from fit_arcfit_exponential_decay import fit_arcfit_exponential_decay
result = fit_arcfit_exponential_decay(<spec_string>, <eval_points>)
print(result)
```

## Inputs
spec_string (str): ARCFIT specification with MODEL, PARAMS, and DATA sections
    eval_points (list): List of x values where to evaluate the fitted model
