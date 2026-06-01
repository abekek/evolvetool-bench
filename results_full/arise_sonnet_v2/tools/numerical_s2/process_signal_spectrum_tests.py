import base64
import struct
import math

def test_valid_arcsig_signal():
    """Test processing a valid ARCSIG signal"""
    # Create test signal: simple sine wave
    sample_rate = 100.0
    samples = [math.sin(2 * math.pi * 5 * t / sample_rate) for t in range(32)]  # 5 Hz sine wave
    
    # Encode as ARCSIG format
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    signal_string = f'ARCSIG:{encoded_data}:{sample_rate}'
    
    result = process_signal_spectrum(signal_string)
    
    # Should return a list of [frequency, magnitude] pairs
    assert isinstance(result, list)
    assert len(result) > 0
    
    # Check structure - each element should be [freq, mag]
    for item in result:
        assert isinstance(item, list)
        assert len(item) == 2
        assert isinstance(item[0], (int, float))  # frequency
        assert isinstance(item[1], (int, float))  # magnitude
        assert item[0] >= 0  # frequencies should be non-negative
        assert item[1] >= 0  # magnitudes should be non-negative
    
    # Frequencies should be in ascending order
    frequencies = [item[0] for item in result]
    assert frequencies == sorted(frequencies)
    
    # First frequency should be 0 (DC)
    assert result[0][0] == 0.0
    
    # Maximum frequency should not exceed Nyquist
    max_freq = max(frequencies)
    assert max_freq <= sample_rate / 2

def test_invalid_format():
    """Test handling of invalid ARCSIG format"""
    result = process_signal_spectrum('INVALID:format')
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for invalid format
    
    result = process_signal_spectrum('ARCSIG:data')  # Missing sample rate
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for invalid format

def test_invalid_sample_rate():
    """Test handling of invalid sample rates"""
    # Create minimal valid data
    samples = [1.0, 0.0, -1.0, 0.0]
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    
    # Test non-numeric sample rate
    signal_string = f'ARCSIG:{encoded_data}:invalid'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for invalid sample rate
    
    # Test negative sample rate
    signal_string = f'ARCSIG:{encoded_data}:-100'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for negative sample rate

def test_invalid_base64():
    """Test handling of invalid base64 data"""
    signal_string = 'ARCSIG:invalid_base64!@#:100'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for invalid base64

def test_empty_signal():
    """Test handling of empty signal data"""
    encoded_data = base64.b64encode(b'').decode('ascii')
    signal_string = f'ARCSIG:{encoded_data}:100'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for empty data

def test_dc_signal():
    """Test processing a DC (constant) signal"""
    # Create DC signal (all samples = 1.0)
    samples = [1.0] * 16
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    signal_string = f'ARCSIG:{encoded_data}:100'
    
    result = process_signal_spectrum(signal_string)
    
    assert isinstance(result, list)
    assert len(result) > 0
    
    # For DC signal, most energy should be at frequency 0
    dc_magnitude = result[0][1]  # Magnitude at 0 Hz
    other_magnitudes = [item[1] for item in result[1:]]
    
    # DC should have the highest magnitude
    assert dc_magnitude > 0
    assert all(dc_magnitude >= mag for mag in other_magnitudes)

def test_power_of_two_length():
    """Test that power-of-2 length signals work correctly"""
    # Create signal with exactly 16 samples (power of 2)
    samples = [math.sin(2 * math.pi * i / 16) for i in range(16)]
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    signal_string = f'ARCSIG:{encoded_data}:16'
    
    result = process_signal_spectrum(signal_string)
    
    assert isinstance(result, list)
    # For 16 samples, one-sided spectrum should have 9 points (0 to 8 inclusive)
    assert len(result) == 9
    
    # Check frequency spacing
    freq_spacing = result[1][0] - result[0][0]
    assert abs(freq_spacing - 1.0) < 1e-10  # Should be 16/16 = 1 Hz

def test_empty_input():
    """Test handling of empty input"""
    result = process_signal_spectrum('')
    assert isinstance(result, list)
    assert len(result) == 0
    
    result = process_signal_spectrum(None)
    assert isinstance(result, list)
    assert len(result) == 0

def test_adversarial_nan_and_inf_samples():
    """Test handling of NaN and infinity values in samples"""
    # Test NaN samples
    samples = [1.0, float('nan'), 3.0, 4.0]
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    signal_string = f'ARCSIG:{encoded_data}:100'
    
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for NaN samples
    
    # Test positive infinity samples
    samples = [1.0, float('inf'), 3.0, 4.0]
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    signal_string = f'ARCSIG:{encoded_data}:100'
    
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for infinity samples
    
    # Test negative infinity samples
    samples = [1.0, float('-inf'), 3.0, 4.0]
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    signal_string = f'ARCSIG:{encoded_data}:100'
    
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for negative infinity samples

def test_strengthened_zero_sample_rate_boundary():
    """Test that sample_rate = 0 is properly rejected (boundary condition for <= vs <)"""
    samples = [1.0, 2.0, 3.0, 4.0]
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    
    # Test exactly zero sample rate
    signal_string = f'ARCSIG:{encoded_data}:0'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for zero sample rate
    
    # Test very small positive sample rate (should work)
    signal_string = f'ARCSIG:{encoded_data}:0.0001'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) > 0  # Should work for positive sample rate

def test_strengthened_negative_zero_sample_rate():
    """Test negative zero and very small negative sample rates"""
    samples = [1.0, 2.0, 3.0, 4.0]
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    
    # Test negative zero
    signal_string = f'ARCSIG:{encoded_data}:-0.0'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for negative zero
    
    # Test very small negative sample rate
    signal_string = f'ARCSIG:{encoded_data}:-0.0001'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for negative sample rate

def test_strengthened_floating_point_zero_sample_rate():
    """Test various floating point representations of zero sample rate"""
    samples = [1.0, 2.0, 3.0, 4.0]
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    
    # Test string "0.0"
    signal_string = f'ARCSIG:{encoded_data}:0.0'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for zero sample rate
    
    # Test string "0.00000"
    signal_string = f'ARCSIG:{encoded_data}:0.00000'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for zero sample rate
    
    # Test scientific notation zero
    signal_string = f'ARCSIG:{encoded_data}:0e10'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should return empty list for zero sample rate

def test_strengthened_sample_rate_boundary_precision():
    """Test sample rate boundary with high precision values"""
    samples = [1.0, 0.0, -1.0, 0.0]
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    
    # Test extremely small positive sample rate
    signal_string = f'ARCSIG:{encoded_data}:1e-100'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) > 0  # Should work for tiny positive sample rate
    
    # Test extremely small negative sample rate
    signal_string = f'ARCSIG:{encoded_data}:-1e-100'
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) == 0  # Should fail for tiny negative sample rate
    
    # Verify the positive case actually produces correct frequency values
    if len(result) > 0:
        frequencies = [item[0] for item in result]
        assert all(freq >= 0 for freq in frequencies)
        assert frequencies[0] == 0.0  # DC component

def test_strengthened_sample_rate_edge_cases_with_validation():
    """Test sample rate validation with edge cases that could expose <= vs < bugs"""
    samples = [1.0, 2.0]  # Minimal valid samples
    binary_data = struct.pack(f'<{len(samples)}d', *samples)
    encoded_data = base64.b64encode(binary_data).decode('ascii')
    
    # Test multiple zero representations
    zero_representations = ['0', '0.', '.0', '0.0', '-0', '-0.0', '+0', '+0.0']
    
    for zero_repr in zero_representations:
        signal_string = f'ARCSIG:{encoded_data}:{zero_repr}'
        result = process_signal_spectrum(signal_string)
        assert isinstance(result, list)
        assert len(result) == 0, f"Failed for zero representation: {zero_repr}"
    
    # Test smallest positive float that should work
    signal_string = f'ARCSIG:{encoded_data}:2.2250738585072014e-308'  # Close to smallest positive float64
    result = process_signal_spectrum(signal_string)
    assert isinstance(result, list)
    assert len(result) > 0  # Should work for smallest positive value
    
    # Verify result structure for the positive case
    for item in result:
        assert len(item) == 2
        assert item[0] >= 0  # frequency >= 0
        assert item[1] >= 0  # magnitude >= 0