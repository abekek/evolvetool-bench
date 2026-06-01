import base64

def test_decode_valid_base64():
    """Test decoding valid base64 data."""
    # Test with known text
    original_text = "Hello, World!"
    encoded = base64.b64encode(original_text.encode('utf-8')).decode('ascii')
    
    result = decode_base64(encoded)
    assert isinstance(result, bytes)
    assert result == original_text.encode('utf-8')
    assert result.decode('utf-8') == original_text

def test_decode_binary_data():
    """Test decoding base64 encoded binary data."""
    # Create some binary data
    original_binary = bytes([0, 1, 2, 3, 255, 254, 253])
    encoded = base64.b64encode(original_binary).decode('ascii')
    
    result = decode_base64(encoded)
    assert isinstance(result, bytes)
    assert result == original_binary

def test_decode_empty_string():
    """Test decoding empty string."""
    result = decode_base64("")
    assert result == b''

def test_decode_whitespace_handling():
    """Test that whitespace is properly handled."""
    original_text = "Test data"
    encoded = base64.b64encode(original_text.encode('utf-8')).decode('ascii')
    
    # Test with leading/trailing whitespace
    result = decode_base64(f"  {encoded}  ")
    assert result == original_text.encode('utf-8')

def test_decode_invalid_base64():
    """Test handling of invalid base64 data."""
    # Invalid characters
    result = decode_base64("Invalid!@#$%^&*()")
    assert result == b''
    
    # Invalid padding
    result = decode_base64("SGVsbG8=")  # Valid, but let's test malformed
    # Actually this is valid, so let's use truly invalid
    result = decode_base64("SGVsbG8!")
    assert result == b''

def test_decode_non_string_input():
    """Test handling of non-string input."""
    result = decode_base64(None)
    assert result == b''
    
    result = decode_base64(123)
    assert result == b''
    
    result = decode_base64([])
    assert result == b''

def test_decode_structured_data():
    """Test decoding base64 that contains structured binary data."""
    import struct
    
    # Create structured binary data (e.g., integers and floats)
    original_data = struct.pack('!IIf', 12345, 67890, 3.14159)
    encoded = base64.b64encode(original_data).decode('ascii')
    
    result = decode_base64(encoded)
    assert isinstance(result, bytes)
    assert result == original_data
    
    # Verify we can unpack it back
    unpacked = struct.unpack('!IIf', result)
    # Check integers exactly and float with tolerance
    assert unpacked[0] == 12345
    assert unpacked[1] == 67890
    assert abs(unpacked[2] - 3.14159) < 1e-5

import base64

def test_independent_basic_base64_decoding():
    """Test basic base64 decoding functionality"""
    # Test simple ASCII string
    original_data = b"Hello, World!"
    encoded = base64.b64encode(original_data).decode('ascii')
    result = decode_base64(encoded)
    assert result == original_data
    assert isinstance(result, bytes)

def test_independent_binary_data_decoding():
    """Test decoding of binary data that could represent QLOG format"""
    # Create binary data that could be part of a quantized log format
    original_binary = bytes([0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD, 0xFC])
    encoded = base64.b64encode(original_binary).decode('ascii')
    result = decode_base64(encoded)
    assert result == original_binary
    assert isinstance(result, bytes)
    assert len(result) == 8

def test_independent_structured_log_record_format():
    """Test decoding data that resembles structured log records with severity levels"""
    # Simulate a simple log record structure: timestamp(4) + severity(1) + message_len(1) + message
    import struct
    timestamp = 1234567890
    severity = 2  # Could represent WARNING level
    message = b"Test log entry"
    message_len = len(message)
    
    log_record = struct.pack('>IB B', timestamp, severity, message_len) + message
    encoded = base64.b64encode(log_record).decode('ascii')
    result = decode_base64(encoded)
    
    assert result == log_record
    assert isinstance(result, bytes)
    # Verify we can extract the components back
    unpacked_timestamp, unpacked_severity, unpacked_len = struct.unpack('>IB B', result[:6])
    assert unpacked_timestamp == timestamp
    assert unpacked_severity == severity
    assert unpacked_len == message_len

def test_independent_empty_and_padding_cases():
    """Test edge cases with empty data and various padding scenarios"""
    # Test empty data
    empty_encoded = base64.b64encode(b"").decode('ascii')
    result = decode_base64(empty_encoded)
    assert result == b""
    assert isinstance(result, bytes)
    
    # Test data that requires padding
    single_byte = b"A"
    encoded_single = base64.b64encode(single_byte).decode('ascii')
    result_single = decode_base64(encoded_single)
    assert result_single == single_byte
    
    # Test data with different padding lengths
    two_bytes = b"AB"
    encoded_two = base64.b64encode(two_bytes).decode('ascii')
    result_two = decode_base64(encoded_two)
    assert result_two == two_bytes

def test_independent_invalid_input_handling():
    """Test error handling for invalid base64 inputs"""
    # Test invalid characters
    try:
        decode_base64("Invalid@#$%Characters!")
        assert False, "Should have raised an exception for invalid characters"
    except Exception:
        pass  # Expected to fail
    
    # Test invalid padding
    try:
        decode_base64("SGVsbG8=====")  # Too much padding
        assert False, "Should have raised an exception for invalid padding"
    except Exception:
        pass  # Expected to fail
    
    # Test non-string input by checking the function signature expectation
    try:
        decode_base64(None)
        assert False, "Should have raised an exception for None input"
    except Exception:
        pass  # Expected to fail
    
    # Test incomplete base64 string
    try:
        decode_base64("SGVsbG")  # Incomplete, missing proper padding
        assert False, "Should have raised an exception for incomplete input"
    except Exception:
        pass  # Expected to fail

def test_adversarial_unicode_and_encoding_edge_cases():
    """Test unicode strings and encoding edge cases that might break base64 decoding."""
    # Unicode string that looks like base64 but contains non-ASCII characters
    unicode_fake_b64 = "SGVsbG8gV29ybGQh"  # Valid base64
    unicode_fake_b64_with_unicode = "SGVsbG8gV29ybGQh" + "ñ"  # Add unicode char
    
    result = decode_base64(unicode_fake_b64_with_unicode)
    assert result == b''  # Should fail gracefully
    
    # Test with unicode whitespace characters (not just ASCII space)
    valid_b64 = "SGVsbG8gV29ybGQh"
    unicode_whitespace_b64 = "\u2000" + valid_b64 + "\u2001"  # En space and em space
    result = decode_base64(unicode_whitespace_b64)
    # This might not be stripped properly if only ASCII whitespace is handled
    assert isinstance(result, bytes)

def test_adversarial_extremely_long_input():
    """Test with extremely long base64 strings to check for resource exhaustion."""
    import base64
    
    # Create a very large binary data (1MB)
    large_data = b'A' * (1024 * 1024)
    encoded_large = base64.b64encode(large_data).decode('ascii')
    
    result = decode_base64(encoded_large)
    assert isinstance(result, bytes)
    assert len(result) == len(large_data)
    
    # Test with malformed extremely long string
    malformed_large = 'A' * (1024 * 1024) + '!'  # Invalid character at end
    result_malformed = decode_base64(malformed_large)
    assert result_malformed == b''

def test_adversarial_base64_padding_manipulation():
    """Test various padding manipulations that might expose edge cases."""
    import base64
    
    # Valid base64 with correct padding
    original = b"Hello"
    valid_encoded = base64.b64encode(original).decode('ascii')  # Should be "SGVsbG8="
    
    # Remove padding entirely
    no_padding = valid_encoded.rstrip('=')
    result_no_padding = decode_base64(no_padding)
    # base64.b64decode with validate=True might be strict about this
    
    # Add extra padding
    extra_padding = valid_encoded + "="
    result_extra = decode_base64(extra_padding)
    assert result_extra == b''  # Should fail with validate=True
    
    # Wrong padding character
    wrong_padding = valid_encoded.replace('=', '-')
    result_wrong = decode_base64(wrong_padding)
    assert result_wrong == b''

def test_adversarial_whitespace_variations():
    """Test various types of whitespace that strip() might not handle correctly."""
    import base64
    
    original = b"Test"
    valid_b64 = base64.b64encode(original).decode('ascii')
    
    # Test with different types of whitespace
    whitespace_variants = [
        f"\t{valid_b64}\t",  # Tabs
        f"\n{valid_b64}\n",  # Newlines
        f"\r{valid_b64}\r",  # Carriage returns
        f"\v{valid_b64}\v",  # Vertical tabs
        f"\f{valid_b64}\f",  # Form feeds
        f" \t\n{valid_b64}\r\v\f ",  # Mixed whitespace
    ]
    
    for variant in whitespace_variants:
        result = decode_base64(variant)
        assert isinstance(result, bytes)
        # The function should handle all these correctly, but let's verify
    
    # Test with whitespace in the middle (should fail)
    middle_whitespace = valid_b64[:4] + " " + valid_b64[4:]
    result_middle = decode_base64(middle_whitespace)
    assert result_middle == b''  # Should fail because whitespace in middle is invalid

def test_adversarial_type_confusion_and_subclasses():
    """Test with string subclasses and objects that might fool isinstance check."""
    import base64
    
    # Create a string subclass
    class FakeString(str):
        def __new__(cls, value):
            return str.__new__(cls, value)
        
        def strip(self):
            # Malicious strip that doesn't actually strip
            return self
    
    original = b"Hello"
    valid_b64 = base64.b64encode(original).decode('ascii')
    
    # Test with string subclass
    fake_string = FakeString(valid_b64)
    result = decode_base64(fake_string)
    # isinstance(fake_string, str) returns True, so this should work
    assert isinstance(result, bytes)
    
    # Test with object that has __str__ method
    class StringLike:
        def __init__(self, value):
            self.value = value
        def __str__(self):
            return self.value
    
    string_like = StringLike(valid_b64)
    result_string_like = decode_base64(string_like)
    assert result_string_like == b''  # Should fail isinstance check
    
    # Test idempotency - calling twice should give same result
    result1 = decode_base64(valid_b64)
    result2 = decode_base64(valid_b64)
    assert result1 == result2
    assert isinstance(result1, bytes)
    assert isinstance(result2, bytes)