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

def test_even_length_median():
    """Test median calculation with even number of elements."""
    data = [1.0, 2.0, 3.0, 4.0]
    result = compute_statistics(data)
    
    assert 'error' not in result
    assert result['median'] == 2.5  # (2+3)/2

def test_single_element():
    """Test statistics with single data point."""
    data = [42.0]
    result = compute_statistics(data)
    
    assert 'error' not in result
    assert result['mean'] == 42.0
    assert result['median'] == 42.0
    assert result['standard deviation'] == 0.0
    assert result['min'] == 42.0
    assert result['max'] == 42.0
    assert result['count'] == 1.0

def test_negative_values():
    """Test with negative values."""
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    result = compute_statistics(data)
    
    assert 'error' not in result
    assert result['mean'] == 0.0
    assert result['median'] == 0.0
    assert result['min'] == -2.0
    assert result['max'] == 2.0

def test_mixed_int_float():
    """Test with mixed integer and float input."""
    data = [1, 2.5, 3, 4.5, 5]
    result = compute_statistics(data)
    
    assert 'error' not in result
    assert result['mean'] == 3.2
    assert result['count'] == 5.0

def test_example_signal_data():
    """Test with the example signal amplitude data from requirements."""
    data = [0.84, -0.54, 0.91, -0.99, 0.14, 0.66, -0.76, 0.96, -0.28, -0.47]
    result = compute_statistics(data)
    
    assert 'error' not in result
    assert result['count'] == 10.0
    
    # Verify mean calculation
    expected_mean = sum(data) / len(data)
    assert abs(result['mean'] - expected_mean) < 1e-10
    
    # Verify median calculation
    sorted_data = sorted(data)
    expected_median = (sorted_data[4] + sorted_data[5]) / 2  # 10 elements, so average of 5th and 6th
    assert abs(result['median'] - expected_median) < 1e-10
    
    # Verify std dev calculation
    variance = sum((x - expected_mean) ** 2 for x in data) / len(data)
    expected_std = math.sqrt(variance)
    assert abs(result['standard deviation'] - expected_std) < 1e-10

def test_empty_data():
    """Test error handling for empty dataset."""
    result = compute_statistics([])
    assert 'error' in result
    assert 'Empty dataset' in result['error']

def test_non_numeric_data():
    """Test error handling for non-numeric data."""
    result = compute_statistics([1, 2, 'invalid', 4])
    assert 'error' in result
    assert 'numeric' in result['error']

def test_identical_values():
    """Test with all identical values."""
    data = [5.0, 5.0, 5.0, 5.0]
    result = compute_statistics(data)
    
    assert 'error' not in result
    assert result['mean'] == 5.0
    assert result['median'] == 5.0
    assert result['standard deviation'] == 0.0
    assert result['min'] == 5.0
    assert result['max'] == 5.0

def test_adversarial_none_input():
    """Test with None as input - should handle gracefully."""
    result = compute_statistics(None)
    assert 'error' in result
    assert isinstance(result, dict)

def test_adversarial_extreme_float_values():
    """Test with extreme float values that could cause overflow/underflow."""
    import sys
    data = [sys.float_info.max, sys.float_info.min, -sys.float_info.max]
    result = compute_statistics(data)
    # Should either work or return error, but not crash
    assert isinstance(result, dict)
    if 'error' not in result:
        # If it succeeds, all values should be finite numbers
        for key, value in result.items():
            if key != 'count':
                assert isinstance(value, (int, float))
                assert not (value != value)  # Check for NaN

def test_adversarial_infinity_and_nan():
    """Test with infinity and NaN values."""
    data = [float('inf'), float('-inf'), float('nan'), 1.0, 2.0]
    result = compute_statistics(data)
    # Function should handle these gracefully - either error or valid results
    assert isinstance(result, dict)
    if 'error' not in result:
        # If no error, check that results are reasonable
        assert 'mean' in result
        assert 'median' in result

def test_adversarial_nested_list_input():
    """Test with nested list structure instead of flat list."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    result = compute_statistics(data)
    assert 'error' in result
    assert isinstance(result, dict)

def test_adversarial_very_large_dataset():
    """Test with extremely large dataset that could cause memory/performance issues."""
    # Create a large dataset that might cause memory issues
    large_size = 10**6
    data = [1.0] * large_size
    result = compute_statistics(data)
    
    # Should complete without crashing
    assert isinstance(result, dict)
    if 'error' not in result:
        assert result['count'] == float(large_size)
        assert result['mean'] == 1.0
        assert result['median'] == 1.0
        assert result['standard deviation'] == 0.0
        assert result['min'] == 1.0
        assert result['max'] == 1.0