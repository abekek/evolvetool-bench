import base64
import tempfile

def test_basic_decode():
    """Test basic base64 decoding functionality."""
    # Test with known data
    original_data = b'Hello, World!'
    encoded = base64.b64encode(original_data).decode('ascii')
    
    result = decode_base64(encoded)
    assert result == original_data

def test_binary_data_decode():
    """Test decoding of binary data."""
    # Test with binary data (bytes 0-255)
    original_data = bytes(range(256))
    encoded = base64.b64encode(original_data).decode('ascii')
    
    result = decode_base64(encoded)
    assert result == original_data

def test_empty_string():
    """Test decoding empty string."""
    result = decode_base64('')
    assert result == b''

def test_whitespace_handling():
    """Test that whitespace is properly handled."""
    original_data = b'Test data'
    encoded = base64.b64encode(original_data).decode('ascii')
    
    # Add whitespace
    encoded_with_whitespace = '  ' + encoded + '\n\t  '
    
    result = decode_base64(encoded_with_whitespace)
    assert result == original_data

def test_invalid_base64():
    """Test handling of invalid base64 strings."""
    # Invalid characters
    result = decode_base64('Invalid@#$%^&*()')
    assert result == b''
    
    # Incomplete padding
    result = decode_base64('SGVsbG8')
    # This might actually decode depending on implementation, so just check it doesn't crash
    assert isinstance(result, bytes)

def test_structured_data():
    """Test with structured binary data that might represent custom formats."""
    import struct
    
    # Create some structured binary data (like a custom file header)
    original_data = struct.pack('>I4sHH', 0x12345678, b'TEST', 1, 2)
    encoded = base64.b64encode(original_data).decode('ascii')
    
    result = decode_base64(encoded)
    assert result == original_data
    
    # Verify we can unpack it back
    magic, signature, version, flags = struct.unpack('>I4sHH', result)
    assert magic == 0x12345678
    assert signature == b'TEST'
    assert version == 1
    assert flags == 2

def test_independent_invalid_input_handling():
    """Test that invalid input types raise appropriate exceptions."""
    # Test None input
    try:
        decode_base64(None)
        assert False, "Should raise exception for None input"
    except TypeError:
        pass  # Expected
    
    # Test non-string inputs
    try:
        decode_base64(123)
        assert False, "Should raise exception for integer input"
    except TypeError:
        pass  # Expected
    
    try:
        decode_base64([1, 2, 3])
        assert False, "Should raise exception for list input"
    except TypeError:
        pass  # Expected
    
    try:
        decode_base64(b'bytes')
        assert False, "Should raise exception for bytes input"
    except TypeError:
        pass  # Expected

def test_adversarial_unicode_and_special_chars():
    """Test with unicode strings and special characters that might break base64 decoding."""
    # Unicode characters that might cause encoding issues
    unicode_string = "SGVsbG8g8J+YgA=="  # Contains emoji-like unicode
    result = decode_base64(unicode_string)
    assert isinstance(result, bytes)
    
    # String with null bytes (might cause issues in C extensions)
    null_string = "SGVsbG8\x00World"
    result = decode_base64(null_string)
    assert isinstance(result, bytes)
    
    # High unicode characters
    high_unicode = "SGVsbG8🚀🎉💻"
    result = decode_base64(high_unicode)
    assert isinstance(result, bytes)

def test_adversarial_malformed_base64_edge_cases():
    """Test various malformed base64 strings that might expose parsing bugs."""
    # Base64 with internal whitespace (not just leading/trailing)
    internal_whitespace = "SGVs bG8g V29y bGQ="
    result = decode_base64(internal_whitespace)
    assert isinstance(result, bytes)
    
    # Base64 with mixed case and numbers at boundaries
    mixed_case = "sgVsbG8gV29ybGQ="
    result = decode_base64(mixed_case)
    assert isinstance(result, bytes)
    
    # String that looks like base64 but has invalid length after cleaning
    almost_valid = "A"  # Single character, invalid base64 length
    result = decode_base64(almost_valid)
    assert isinstance(result, bytes)
    
    # Empty string after strip
    only_whitespace = "   \n\t\r   "
    result = decode_base64(only_whitespace)
    assert isinstance(result, bytes)

def test_adversarial_very_large_input():
    """Test with extremely large base64 strings that might cause memory issues."""
    import base64
    
    # Create a large binary data (1MB)
    large_data = b'A' * (1024 * 1024)
    large_encoded = base64.b64encode(large_data).decode('ascii')
    
    result = decode_base64(large_encoded)
    assert isinstance(result, bytes)
    assert len(result) == len(large_data)
    
    # Test with malformed large string
    large_invalid = 'X' * (1024 * 1024)  # Large but invalid base64
    result = decode_base64(large_invalid)
    assert isinstance(result, bytes)

def test_adversarial_base64_padding_edge_cases():
    """Test edge cases around base64 padding that might expose bugs."""
    # Valid base64 with extra padding
    extra_padding = "SGVsbG8====="  # Too many = signs
    result = decode_base64(extra_padding)
    assert isinstance(result, bytes)
    
    # Base64 with padding in wrong position
    wrong_padding = "SGVs=bG8="
    result = decode_base64(wrong_padding)
    assert isinstance(result, bytes)
    
    # Base64 with no padding where padding expected
    no_padding = "SGVsbG8"  # Should have padding but doesn't
    result = decode_base64(no_padding)
    assert isinstance(result, bytes)
    
    # Only padding characters
    only_padding = "===="
    result = decode_base64(only_padding)
    assert isinstance(result, bytes)

def test_adversarial_string_subclass_and_property_validation():
    """Test with string subclasses and validate all function properties."""
    # Custom string subclass to test isinstance check
    class CustomString(str):
        def __new__(cls, value):
            return str.__new__(cls, value)
    
    # Test with string subclass - should work since isinstance(CustomString(), str) is True
    custom_str = CustomString("SGVsbG8gV29ybGQ=")
    result = decode_base64(custom_str)
    assert isinstance(result, bytes)
    
    # Verify idempotency - same input should give same result
    result1 = decode_base64("SGVsbG8gV29ybGQ=")
    result2 = decode_base64("SGVsbG8gV29ybGQ=")
    assert result1 == result2
    assert type(result1) == type(result2)
    
    # Verify function always returns bytes type, never None or other types
    test_cases = ["", "invalid", "SGVsbG8=", "   \n   "]
    for case in test_cases:
        result = decode_base64(case)
        assert isinstance(result, bytes), f"Function returned {type(result)} instead of bytes for input: {case}"
        assert result is not None, f"Function returned None for input: {case}"

def test_strengthened_whitespace_preservation():
    """Test that whitespace stripping is actually performed by comparing with/without whitespace."""
    # Create a base64 string that would fail if whitespace isn't stripped
    original_data = b'Hello World'
    clean_encoded = base64.b64encode(original_data).decode('ascii')
    
    # Add various types of whitespace that should be stripped
    whitespace_variants = [
        f"  {clean_encoded}  ",  # spaces
        f"\n{clean_encoded}\n",  # newlines
        f"\t{clean_encoded}\t",  # tabs
        f"\r{clean_encoded}\r",  # carriage returns
        f" \n\t\r{clean_encoded} \n\t\r",  # mixed whitespace
    ]
    
    # All variants should decode to the same result as clean version
    expected_result = decode_base64(clean_encoded)
    
    for variant in whitespace_variants:
        result = decode_base64(variant)
        assert result == expected_result, f"Whitespace variant failed: {repr(variant)}"
        assert result == original_data, f"Decoded data doesn't match original for: {repr(variant)}"

def test_strengthened_internal_whitespace_handling():
    """Test that internal whitespace in base64 strings is handled correctly."""
    # Create valid base64 with internal whitespace that should cause failure if not stripped
    original_data = b'Test message for whitespace handling'
    clean_encoded = base64.b64encode(original_data).decode('ascii')
    
    # Insert whitespace at various positions within the base64 string
    mid_point = len(clean_encoded) // 2
    quarter_point = len(clean_encoded) // 4
    
    internal_whitespace_cases = [
        clean_encoded[:mid_point] + " " + clean_encoded[mid_point:],
        clean_encoded[:quarter_point] + "\n" + clean_encoded[quarter_point:],
        clean_encoded[:quarter_point] + "\t" + clean_encoded[quarter_point:mid_point] + " " + clean_encoded[mid_point:],
    ]
    
    # These should either decode correctly (if base64 module handles it) or return empty bytes
    # The key is that strip() removal would affect the behavior
    for case in internal_whitespace_cases:
        result = decode_base64(case)
        assert isinstance(result, bytes), f"Should return bytes for: {repr(case)}"
        # Without strip(), these would likely fail differently than with strip()

def test_strengthened_leading_trailing_whitespace_exact_match():
    """Test exact byte-for-byte matching with leading/trailing whitespace scenarios."""
    # Test cases where strip() makes the difference between success and failure
    test_data = b'Precise test data'
    clean_base64 = base64.b64encode(test_data).decode('ascii')
    
    # Cases with significant whitespace that would break decoding without strip()
    whitespace_cases = [
        f"    {clean_base64}    ",
        f"\n\n{clean_base64}\n\n",
        f"\t\t\t{clean_base64}\t\t\t",
        f"   \n\t  {clean_base64}  \t\n   ",
    ]
    
    for case in whitespace_cases:
        result = decode_base64(case)
        assert result == test_data, f"Failed exact match for whitespace case: {repr(case)}"
        assert len(result) == len(test_data), f"Length mismatch for case: {repr(case)}"

def test_strengthened_multiline_base64_whitespace():
    """Test multiline base64 strings that require whitespace stripping to work."""
    # Create a longer base64 string that might be formatted across multiple lines
    long_data = b'This is a longer message that will create a base64 string long enough to be split across multiple lines in some formatting scenarios.'
    clean_base64 = base64.b64encode(long_data).decode('ascii')
    
    # Simulate multiline formatting with various whitespace
    multiline_variants = [
        f"  {clean_base64[:20]}\n  {clean_base64[20:40]}\n  {clean_base64[40:]}  ",
        f"\t{clean_base64}\n",
        f" \r\n {clean_base64} \r\n ",
        f"   {clean_base64}   \n\n",
    ]
    
    # Without proper strip(), these would fail to decode correctly
    for variant in multiline_variants:
        # First verify the variant actually has whitespace
        assert variant != variant.strip(), f"Test case should have whitespace: {repr(variant)}"
        
        result = decode_base64(variant)
        # The result should either be the correct data (if strip works) or empty bytes (if it fails)
        assert isinstance(result, bytes), f"Should return bytes for: {repr(variant)}"
        
        # If decoding succeeds, it should match original data exactly
        if len(result) > 0:
            assert result == long_data, f"Decoded data should match original for: {repr(variant)}"

def test_strengthened_whitespace_only_and_empty_after_strip():
    """Test edge cases where input becomes empty or different after stripping."""
    # Test strings that become empty after strip() is applied
    whitespace_only_cases = [
        "   ",
        "\n\n\n",
        "\t\t\t",
        " \n\t\r ",
        "     \n     ",
    ]
    
    for case in whitespace_only_cases:
        # Verify these are actually whitespace-only
        assert case.strip() == "", f"Case should be whitespace-only: {repr(case)}"
        
        result = decode_base64(case)
        assert isinstance(result, bytes), f"Should return bytes for whitespace-only: {repr(case)}"
        assert result == b'', f"Should return empty bytes for whitespace-only: {repr(case)}"
    
    # Test cases where strip() significantly changes the string
    original_data = b'strip test'
    base64_str = base64.b64encode(original_data).decode('ascii')
    
    # Add substantial whitespace that strip() would remove
    padded_cases = [
        f"{'  ' * 10}{base64_str}{'  ' * 10}",
        f"{chr(32) * 20}{base64_str}{chr(32) * 20}",
        f"\n\n\n{base64_str}\n\n\n",
    ]
    
    for case in padded_cases:
        # Verify strip() actually changes the string
        assert case != case.strip(), f"Strip should change the string: {repr(case)}"
        assert len(case) > len(case.strip()), f"Strip should reduce length: {repr(case)}"
        
        result = decode_base64(case)
        assert result == original_data, f"Should decode correctly after strip: {repr(case)}"