# compute_statistics

## Description

Compute basic statistical measures for a numerical dataset.

## Usage

```python
from compute_statistics import compute_statistics
result = compute_statistics(<data>)
print(result)
```

## Inputs
data: List of numerical values to analyze

## Example Tests
```python
import math

def test_basic_statistics():
    """Test basic statistical computation with known values."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = compute_statistics(data)
    
    assert 'error' not in result
    assert result['mean'] == 3.0
    assert result['median'] == 3.0
    assert result['min'] == 1.0
    assert result['max'] == 5.0
    assert result['count'] == 5.0
    
    # Standard deviation for [1,2,3,4,5]: sqrt(sum((x-3)^2)/5) = sqrt(10/5) = sqrt(2)
    expected_std = math.sqrt(2.0)
    assert abs(result['standard deviation'] - expected_std) < 1e-10

def test_even_length_
```
