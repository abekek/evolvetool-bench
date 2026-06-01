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
    assert 'parameters' in result
    assert 'r_squared' in result
    assert 'residual_sum_squares' in result
    assert result['r_squared'] > 0.9  # Should fit well
    
    # Parameters should be reasonably close to true values
    fitted_params = result['parameters']
    assert abs(fitted_params['a'] - true_a) < 0.5
    assert abs(fitted_params['b'] - true_b) < 0.3
    assert abs(fitted_params['c'] - true_c) < 0.3

def test_power_law_fitting():
    """Test fitting a power law model to synthetic data."""
    # Generate synthetic power law data: y = 3 * x^0.8 + 0.5
    true_a, true_b, true_c = 3.0, 0.8, 0.5
    data_points = []
    for i in range(1, 11):
        x = i * 0.5
        y = true_a * (x ** true_b) + true_c
        data_points.append([x, y])
    
    initial_params = {'a': 2.5, 'b': 0.6, 'c': 0.3}
    result = fit_nonlinear_curve('power', data_points, initial_params)
    
    assert result['success'] == True
    assert result['r_squared'] > 0.8
    assert result['residual_sum_squares'] >= 0

def test_gaussian_fitting():
    """Test fitting a Gaussian model to synthetic data."""
    # Generate synthetic Gaussian data: y = 5 * exp(-0.5 * ((x - 2) / 1)^2) + 1
    true_a, true_mu, true_sigma, true_c = 5.0, 2.0, 1.0, 1.0
    data_points = []
    for i in range(21):
        x = i * 0.2
        y = true_a * math.exp(-0.5 * ((x - true_mu) / true_sigma) ** 2) + true_c
        data_points.append([x, y])
    
    initial_params = {'a': 4.0, 'mu': 1.8, 'sigma': 0.8, 'c': 0.8}
    result = fit_nonlinear_curve('gaussian', data_points, initial_params)
    
    assert result['success'] == True
    assert result['r_squared'] > 0.8

def test_invalid_model_type():
    """Test handling of invalid model type."""
    data_points = [[0, 1], [1, 2], [2, 3]]
    initial_params = {'a': 1, 'b': 1, 'c': 1}
    
    try:
        result = fit_nonlinear_curve('invalid_model', data_points, initial_params)
        # If we get here, the function returned instead of raising
        assert result['success'] == False
        assert 'error' in result
        assert 'Unknown model type' in result['error']
    except ValueError as e:
        # This is also acceptable - function raised ValueError
        assert 'Unknown model type' in str(e)

def test_insufficient_data():
    """Test handling of insufficient data points."""
    data_points = [[0, 1]]  # Only one point
    initial_params = {'a': 1, 'b': 1, 'c': 1}
    
    result = fit_nonlinear_curve('exponential_decay', data_points, initial_params)
    
    assert result['success'] == False
    assert 'error' in result
    assert 'at least 2 data points' in result['error']

def test_missing_parameters():
    """Test handling of missing required parameters."""
    data_points = [[0, 1], [1, 2], [2, 3]]
    initial_params = {'a': 1, 'b': 1}  # Missing 'c'
    
    result = fit_nonlinear_curve('exponential_decay', data_points, initial_params)
    
    assert result['success'] == False
    assert 'error' in result
    assert 'Missing required parameters' in result['error']

def test_malformed_data_points():
    """Test handling of malformed data points."""
    data_points = [[0, 1], [1], [2, 3, 4]]  # Inconsistent dimensions
    initial_params = {'a': 1, 'b': 1, 'c': 1}
    
    result = fit_nonlinear_curve('exponential_decay', data_points, initial_params)
    
    assert result['success'] == False
    assert 'error' in result
    assert '[x, y] pairs' in result['error']

def test_independent_invalid_model_type_error():
    """Test that invalid model type raises appropriate exception."""
    data_points = [[0, 1], [1, 2], [2, 3]]
    initial_params = {'a': 1, 'b': 1, 'c': 1}
    
    import pytest
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        fit_nonlinear_curve('invalid_model', data_points, initial_params)

def test_adversarial_empty_and_none_inputs():
    """Test edge cases with empty and None inputs."""
    # Test empty data_points list
    result = fit_nonlinear_curve('exponential_decay', [], {'a': 1, 'b': 1, 'c': 1})
    assert result['success'] == False
    assert 'error' in result
    
    # Test None data_points
    try:
        result = fit_nonlinear_curve('exponential_decay', None, {'a': 1, 'b': 1, 'c': 1})
        assert result['success'] == False
    except (TypeError, AttributeError):
        pass  # Either exception is acceptable
    
    # Test None initial_params
    try:
        result = fit_nonlinear_curve('exponential_decay', [[0, 1], [1, 2]], None)
        assert result['success'] == False
    except (TypeError, AttributeError):
        pass
    
    # Test empty initial_params dict
    result = fit_nonlinear_curve('exponential_decay', [[0, 1], [1, 2]], {})
    assert result['success'] == False
    assert 'Missing required parameters' in result['error']

def test_adversarial_extreme_numeric_values():
    """Test handling of extreme numeric values that could cause overflow/underflow."""
    import math
    
    # Test with extremely large values
    large_data = [[1e10, 1e15], [2e10, 2e15], [3e10, 3e15]]
    large_params = {'a': 1e20, 'b': 1e5, 'c': 1e10}
    result = fit_nonlinear_curve('exponential_growth', large_data, large_params)
    # Should either succeed or fail gracefully
    assert isinstance(result, dict)
    assert 'success' in result
    
    # Test with extremely small values near zero
    tiny_data = [[1e-15, 1e-20], [2e-15, 2e-20], [3e-15, 3e-20]]
    tiny_params = {'a': 1e-10, 'b': 1e-5, 'c': 1e-15}
    result = fit_nonlinear_curve('exponential_decay', tiny_data, tiny_params)
    assert isinstance(result, dict)
    assert 'success' in result
    
    # Test with infinity and NaN values
    inf_data = [[1, float('inf')], [2, 3], [3, 4]]
    result = fit_nonlinear_curve('exponential_decay', inf_data, {'a': 1, 'b': 1, 'c': 1})
    # Should handle gracefully without crashing
    assert isinstance(result, dict)
    
    nan_data = [[float('nan'), 1], [2, 3], [3, 4]]
    result = fit_nonlinear_curve('exponential_decay', nan_data, {'a': 1, 'b': 1, 'c': 1})
    assert isinstance(result, dict)

def test_adversarial_type_confusion():
    """Test type boundary cases and wrong types."""
    # Test with string coordinates that look like numbers
    string_data = [["1", "2"], ["3", "4"], ["5", "6"]]
    result = fit_nonlinear_curve('exponential_decay', string_data, {'a': 1, 'b': 1, 'c': 1})
    # Should either convert successfully or fail gracefully
    assert isinstance(result, dict)
    assert 'success' in result
    
    # Test with mixed types in data points
    mixed_data = [[1, 2], ["3", 4], [5.5, "6"]]
    result = fit_nonlinear_curve('power', mixed_data, {'a': 1, 'b': 1, 'c': 1})
    assert isinstance(result, dict)
    
    # Test with string parameters
    string_params = {'a': "1.5", 'b': "0.5", 'c': "1.0"}
    result = fit_nonlinear_curve('exponential_decay', [[1, 2], [2, 3]], string_params)
    assert isinstance(result, dict)
    
    # Test with boolean values in data
    bool_data = [[True, False], [1, 0], [2, 1]]
    result = fit_nonlinear_curve('exponential_decay', bool_data, {'a': 1, 'b': 1, 'c': 1})
    assert isinstance(result, dict)
    
    # Test with complex numbers
    try:
        complex_data = [[1+2j, 3], [2, 4+1j]]
        result = fit_nonlinear_curve('exponential_decay', complex_data, {'a': 1, 'b': 1, 'c': 1})
        assert isinstance(result, dict)
    except (TypeError, ValueError):
        pass  # Complex numbers should be rejected

def test_adversarial_power_model_zero_negative():
    """Test power model with zero and negative x values that could cause domain errors."""
    # Test power model with zero x values (should use fallback value c)
    zero_data = [[0, 5], [1, 3], [2, 2]]
    result = fit_nonlinear_curve('power', zero_data, {'a': 2, 'b': 0.5, 'c': 1})
    assert isinstance(result, dict)
    assert 'success' in result
    
    # Test power model with negative x values
    negative_data = [[-2, 5], [-1, 3], [1, 2], [2, 1]]
    result = fit_nonlinear_curve('power', negative_data, {'a': 2, 'b': 0.5, 'c': 1})
    assert isinstance(result, dict)
    
    # Test power model with fractional exponent and negative base (should cause issues)
    negative_frac_data = [[-1, 2], [1, 3], [4, 5]]
    result = fit_nonlinear_curve('power', negative_frac_data, {'a': 1, 'b': 0.5, 'c': 0})
    assert isinstance(result, dict)
    # Should either handle gracefully or report error
    
    # Test with zero sigma in Gaussian (division by zero protection)
    gauss_data = [[1, 2], [2, 3], [3, 2]]
    zero_sigma_params = {'a': 1, 'mu': 2, 'sigma': 0, 'c': 1}
    result = fit_nonlinear_curve('gaussian', gauss_data, zero_sigma_params)
    assert isinstance(result, dict)
    assert 'success' in result

def test_adversarial_idempotency_and_consistency():
    """Test that function calls are idempotent and results are consistent."""
    import copy
    
    # Test data and parameters
    test_data = [[0.5, 2.1], [1.0, 1.8], [1.5, 1.4], [2.0, 1.1]]
    test_params = {'a': 2.0, 'b': 0.5, 'c': 1.0}
    
    # Call function multiple times with identical inputs
    result1 = fit_nonlinear_curve('exponential_decay', test_data, test_params)
    result2 = fit_nonlinear_curve('exponential_decay', test_data, test_params)
    result3 = fit_nonlinear_curve('exponential_decay', copy.deepcopy(test_data), copy.deepcopy(test_params))
    
    # Results should be identical
    assert result1['success'] == result2['success'] == result3['success']
    
    if result1['success']:
        # Check that fitted parameters are identical
        for param in result1['parameters']:
            assert abs(result1['parameters'][param] - result2['parameters'][param]) < 1e-10
            assert abs(result1['parameters'][param] - result3['parameters'][param]) < 1e-10
        
        # Check that metrics are identical
        assert abs(result1['r_squared'] - result2['r_squared']) < 1e-10
        assert abs(result1['residual_sum_squares'] - result2['residual_sum_squares']) < 1e-10
    
    # Test that modifying input after call doesn't affect previous results
    original_r_squared = result1.get('r_squared', None)
    test_data[0][1] = 999  # Modify original data
    test_params['a'] = 999  # Modify original params
    
    # Previous results should be unchanged
    if original_r_squared is not None:
        assert result1['r_squared'] == original_r_squared
    
    # Test return value structure consistency
    expected_keys_success = {'success', 'parameters', 'r_squared', 'residual_sum_squares'}
    expected_keys_failure = {'success', 'error'}
    
    if result1['success']:
        assert set(result1.keys()) == expected_keys_success
        assert isinstance(result1['parameters'], dict)
        assert isinstance(result1['r_squared'], (int, float))
        assert isinstance(result1['residual_sum_squares'], (int, float))
    else:
        assert set(result1.keys()) == expected_keys_failure
        assert isinstance(result1['error'], str)