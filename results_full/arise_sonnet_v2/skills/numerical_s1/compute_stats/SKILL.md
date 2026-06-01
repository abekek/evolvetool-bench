# compute_stats

## Description

Calculate basic statistical measures for a numerical dataset.

## Usage

```python
from compute_stats import compute_stats
result = compute_stats(<data>)
print(result)
```

## Inputs
data: List of numerical values

## Example Tests
```python
import math

def test_basic_statistics():
    """Test basic statistical calculations with known values."""
    data = [1, 2, 3, 4, 5]
    result = compute_stats(data)
    
    assert "error" not in result
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["count"] == 5
    
    # Standard deviation: sqrt(sum((x-mean)^2)/n) = sqrt(10/5) = sqrt(2)
    expected_std = math.sqrt(2)
    assert abs(result["standard deviation"] - expected_std) < 1e-10

def test_even_length_median():
    """Test median ca
```
