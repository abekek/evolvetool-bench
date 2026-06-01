# fit_exponential_decay_robust

## Description

Robust exponential decay fitting for noisy data with small signal amplitude.

## Usage

```python
from fit_exponential_decay_robust import fit_exponential_decay_robust
result = fit_exponential_decay_robust(<data_string>, <convergence_tolerance>, <max_iterations>)
print(result)
```

## Inputs
data_string (str): Pipe-separated data points in format "x1,y1|x2,y2|..."
    convergence_tolerance (float): Convergence threshold for optimization
    max_iterations (int): Maximum iterations for optimization algorithm
