# get_unix_timestamp

## Description

Get the current Unix timestamp as an integer.

## Usage

```python
from get_unix_timestamp import get_unix_timestamp
result = get_unix_timestamp()
print(result)
```

## Returns
int: Current Unix timestamp in seconds since epoch

## Example Tests
```python
import time
import datetime

def test_returns_integer():
    """Test that the function returns an integer."""
    result = get_unix_timestamp()
    assert isinstance(result, int), f"Expected int, got {type(result)}"

def test_returns_positive_value():
    """Test that the function returns a positive timestamp."""
    result = get_unix_timestamp()
    assert result > 0, f"Expected positive timestamp, got {result}"

def test_reasonable_timestamp_range():
    """Test that timestamp is in a reasonable range (after 2020, before 2050)."""
    result = get_unix_timestamp()
    # January 1, 2020 UTC
 
```
