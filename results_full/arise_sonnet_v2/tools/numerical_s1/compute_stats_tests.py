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
    """Test median calculation with even number of elements."""
    data = [1, 2, 3, 4]
    result = compute_stats(data)
    
    assert "error" not in result
    assert result["median"] == 2.5  # (2 + 3) / 2

def test_single_element():
    """Test statistics for single element dataset."""
    data = [42.5]
    result = compute_stats(data)
    
    assert "error" not in result
    assert result["mean"] == 42.5
    assert result["median"] == 42.5
    assert result["standard deviation"] == 0.0
    assert result["min"] == 42.5
    assert result["max"] == 42.5
    assert result["count"] == 1

def test_mixed_int_float():
    """Test with mixed integer and float values."""
    data = [1, 2.5, 3, 4.5]
    result = compute_stats(data)
    
    assert "error" not in result
    assert result["mean"] == 2.75
    assert result["median"] == 2.75
    assert result["count"] == 4

def test_empty_dataset():
    """Test error handling for empty dataset."""
    result = compute_stats([])
    assert "error" in result
    assert "Empty dataset" in result["error"]

def test_non_numerical_data():
    """Test error handling for non-numerical data."""
    result = compute_stats([1, 2, "three", 4])
    assert "error" in result
    assert "numerical" in result["error"]

def test_example_dataset():
    """Test with the example dataset from requirements."""
    data = [4.2, 7.8, 3.1, 9.5, 2.6, 6.4, 8.3, 1.9, 5.7, 4.8]
    result = compute_stats(data)
    
    assert "error" not in result
    assert result["count"] == 10
    
    # Verify mean calculation
    expected_mean = sum(data) / len(data)
    assert abs(result["mean"] - expected_mean) < 1e-10
    
    # Verify median (middle of sorted list)
    sorted_data = sorted(data)
    expected_median = (sorted_data[4] + sorted_data[5]) / 2
    assert abs(result["median"] - expected_median) < 1e-10
    
    # Verify min/max
    assert result["min"] == min(data)
    assert result["max"] == max(data)

def test_adversarial_extreme_values():
    """Test with extreme values that might cause numerical issues."""
    large_val = 1e15
    small_val = 1e-15
    data = [large_val, small_val]
    result = compute_stats(data)
    
    assert "error" not in result
    
    # For extreme values, use relative tolerance for comparison
    expected_mean = (large_val + small_val) / 2
    assert abs(result["mean"] - expected_mean) / max(abs(expected_mean), 1) < 1e-10
    
    assert result["min"] == small_val
    assert result["max"] == large_val
    assert result["count"] == 2

def test_invalid_values():
    """Test error handling for NaN and infinity values."""
    # Test NaN
    result = compute_stats([1, 2, float('nan'), 4])
    assert "error" in result
    assert "non-finite" in result["error"]
    
    # Test infinity
    result = compute_stats([1, 2, float('inf'), 4])
    assert "error" in result
    assert "non-finite" in result["error"]
    
    # Test negative infinity
    result = compute_stats([1, 2, float('-inf'), 4])
    assert "error" in result
    assert "non-finite" in result["error"]

def test_identical_values():
    """Test with dataset containing identical values."""
    data = [5.0, 5.0, 5.0, 5.0]
    result = compute_stats(data)
    
    assert "error" not in result
    assert result["mean"] == 5.0
    assert result["median"] == 5.0
    assert result["standard deviation"] == 0.0
    assert result["min"] == 5.0
    assert result["max"] == 5.0
    assert result["count"] == 4