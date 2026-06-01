# fit_exponential_decay_arcfit

## Description

Fits an exponential decay model to noisy data with robust handling of near-degenerate cases.

## Usage

```python
from fit_exponential_decay_arcfit import fit_exponential_decay_arcfit
result = fit_exponential_decay_arcfit(<data_string>, <tolerance>)
print(result)
```

## Inputs
data_string (str): Pipe-separated data points in format "x1,y1|x2,y2|..."
    tolerance (float): Acceptable tolerance for offset parameter validation
