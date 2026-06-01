import struct
import tempfile

def test_basic_parsing():
    """Test basic struct format parsing"""
    # Create test data: big-endian uint32 (value 0x12345678)
    test_data = struct.pack('>I', 0x12345678)
    result = parse_binary_data(test_data, '>I')
    assert result == [0x12345678]

def test_multiple_values():
    """Test parsing multiple values in one format"""
    # Create test data: two little-endian uint16s
    test_data = struct.pack('<HH', 0x1234, 0x5678)
    result = parse_binary_data(test_data, '<HH')
    assert result == [0x1234, 0x5678]

def test_sequential_parsing():
    """Test sequential parsing with comma-separated formats"""
    # Create test data: big-endian uint32 followed by unsigned char
    test_data = struct.pack('>I', 0x12345678) + struct.pack('B', 0xAB)
    result = parse_binary_data(test_data, '>I,B')
    assert result == [0x12345678, 0xAB]

def test_complex_format():
    """Test complex format with mixed endianness"""
    # Big-endian uint32, little-endian uint16, unsigned char
    part1 = struct.pack('>I', 0x12345678)
    part2 = struct.pack('<H', 0x9ABC)
    part3 = struct.pack('B', 0xDE)
    test_data = part1 + part2 + part3
    result = parse_binary_data(test_data, '>I,<H,B')
    assert result == [0x12345678, 0x9ABC, 0xDE]

def test_insufficient_data():
    """Test error handling for insufficient data"""
    test_data = b'\x12\x34'  # Only 2 bytes
    result = parse_binary_data(test_data, '>I')  # Needs 4 bytes
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'Not enough data' in result[0]['error']

def test_invalid_format():
    """Test error handling for invalid format string"""
    test_data = b'\x12\x34\x56\x78'
    result = parse_binary_data(test_data, 'X')  # Invalid format character
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'format error' in result[0]['error'].lower()

def test_non_bytes_input():
    """Test error handling for non-bytes input"""
    result = parse_binary_data("not bytes", '>I')
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'must be bytes' in result[0]['error']

def test_empty_format():
    """Test error handling for empty format specification"""
    test_data = b'\x12\x34\x56\x78'
    result = parse_binary_data(test_data, '')
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'cannot be empty' in result[0]['error']

def test_signed_values():
    """Test parsing signed integer values"""
    # Create test data with negative signed int
    test_data = struct.pack('>i', -1000)  # Signed int
    result = parse_binary_data(test_data, '>i')
    assert result == [-1000]

def test_float_values():
    """Test parsing float values"""
    # Create test data with float
    test_value = 3.14159
    test_data = struct.pack('>f', test_value)
    result = parse_binary_data(test_data, '>f')
    assert len(result) == 1
    assert abs(result[0] - test_value) < 0.0001  # Float precision comparison

def test_adversarial_none_inputs():
    """Test function behavior with None inputs"""
    # Test None data
    result = parse_binary_data(None, '>I')
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'error' in result[0]
    
    # Test None format_spec
    test_data = b'\x12\x34\x56\x78'
    result = parse_binary_data(test_data, None)
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'error' in result[0]

def test_adversarial_malicious_format_strings():
    """Test with potentially malicious or malformed format strings"""
    test_data = b'\x12\x34\x56\x78' * 100
    
    # Test with extremely long format string
    malicious_format = 'B' * 10000
    result = parse_binary_data(test_data, malicious_format)
    assert isinstance(result, list)
    
    # Test with format containing only commas and whitespace
    result = parse_binary_data(test_data, ',,,   ,  ,')
    assert isinstance(result, list)
    assert result == []  # Should return empty list since all parts are empty after strip
    
    # Test with format containing special characters that might cause issues
    result = parse_binary_data(test_data, '>I\x00\xff')
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'error' in result[0]

def test_adversarial_resource_exhaustion():
    """Test potential resource exhaustion scenarios"""
    # Create large binary data
    large_data = b'\x00' * (1024 * 1024)  # 1MB of zeros
    
    # Test with format that would create many small extractions
    format_parts = ['B'] * 1000  # Try to extract 1000 individual bytes
    large_format = ','.join(format_parts)
    result = parse_binary_data(large_data, large_format)
    assert isinstance(result, list)
    assert len(result) == 1000  # Should successfully parse 1000 bytes
    
    # Test with empty data but complex format
    result = parse_binary_data(b'', large_format)
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'Not enough data' in result[0]['error']

def test_adversarial_boundary_data_sizes():
    """Test edge cases around data size boundaries"""
    # Test with exactly the right amount of data
    test_data = b'\x12\x34\x56\x78'
    result = parse_binary_data(test_data, '>I')
    assert result == [0x12345678]
    
    # Test with one byte less than needed
    test_data = b'\x12\x34\x56'
    result = parse_binary_data(test_data, '>I')
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'error' in result[0]
    
    # Test with empty bytes
    result = parse_binary_data(b'', 'B')
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'error' in result[0]
    
    # Test sequential parsing where first succeeds but second fails
    test_data = b'\x12\x34\x56\x78\x9A'  # 5 bytes
    result = parse_binary_data(test_data, '>I,>I')  # Need 4+4=8 bytes total
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'offset 4' in result[0]['error']

def test_adversarial_type_confusion():
    """Test type confusion and unexpected input types"""
    # Test with bytearray instead of bytes
    test_data = bytearray([0x12, 0x34, 0x56, 0x78])
    result = parse_binary_data(test_data, '>I')
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'must be bytes' in result[0]['error']
    
    # Test with memoryview
    test_bytes = b'\x12\x34\x56\x78'
    test_data = memoryview(test_bytes)
    result = parse_binary_data(test_data, '>I')
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'error' in result[0]
    
    # Test with integer as format_spec
    test_data = b'\x12\x34\x56\x78'
    result = parse_binary_data(test_data, 123)
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'error' in result[0]
    
    # Test idempotency - same input should give same result
    test_data = b'\x12\x34\x56\x78\x9A\xBC'
    format_spec = '>I,<H'
    result1 = parse_binary_data(test_data, format_spec)
    result2 = parse_binary_data(test_data, format_spec)
    assert result1 == result2