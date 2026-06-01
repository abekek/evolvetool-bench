# fit_nonlinear_curve

## Description

Perform non-linear curve fitting using least squares optimization.

## Usage

```python
from fit_nonlinear_curve import fit_nonlinear_curve
result = fit_nonlinear_curve(<model_type>, <data_points>, <initial_params>)
print(result)
```

## Inputs
model_type: Type of model to fit ('exponential_decay', 'exponential_growth', 'power', 'gaussian')
    data_points: List of [x, y] coordinate pairs
    initial_params: Dictionary of initial parameter guesses for the model

## Example Tests
```python
import math

def test_exponential_decay_fitting():
    """Test fitting an exponential decay model to synthetic data."""
    # Generate synthetic exponential decay data: y = 2 * exp(-0.5 * x) + 1
    true_a, true_b, true_c = 2.0, 0.5, 1.0
    data_points = []
    for i in range(10):
        x = i * 0.5
        y = true_a * math.exp(-true_b * x) + true_c
        data_points.append([x, y])
    
    initial_params = {'a': 1.5, 'b': 0.3, 'c': 0.8}
    result = fit_nonlinear_curve('exponential_decay', data_points, initial_params)
    
    assert result['success'] == True
    assert 'parameters' in r
```
