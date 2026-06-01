import base64

def test_decode_abr_basic():
    """Test basic ABR decoding with simple key-value pairs."""
    # Create test data: record with city=NYC, temp=72
    data = bytearray()
    # First record
    data.extend([4])  # key length
    data.extend(b'city')  # key
    data.extend([3])  # value length  
    data.extend(b'NYC')  # value
    data.extend([4])  # key length
    data.extend(b'temp')  # key
    data.extend([2])  # value length
    data.extend(b'72')  # value
    data.extend([0xFF])  # record separator
    # Second record
    data.extend([4])  # key length
    data.extend(b'city')  # key
    data.extend([2])  # value length
    data.extend(b'LA')  # value
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    assert len(result) == 2
    assert result[0]['city'] == 'NYC'
    assert result[0]['temp'] == '72'
    assert result[1]['city'] == 'LA'

def test_decode_abr_empty_input():
    """Test ABR decoding with empty input."""
    result = decode_abr_format('')
    assert isinstance(result, list)
    assert len(result) == 0
    
    result = decode_abr_format('   ')
    assert isinstance(result, list)
    assert len(result) == 0

def test_decode_abr_invalid_base64():
    """Test ABR decoding with invalid base64."""
    result = decode_abr_format('invalid_base64!')
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'Failed to decode ABR format' in result[0]['error']

def test_decode_abr_single_record():
    """Test ABR decoding with single record."""
    data = bytearray()
    data.extend([3])  # key length
    data.extend(b'key')  # key
    data.extend([5])  # value length
    data.extend(b'value')  # value
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    assert len(result) == 1
    assert result[0]['key'] == 'value'

def test_decode_abr_multiple_separators():
    """Test ABR decoding with multiple 0xFF separators."""
    data = bytearray()
    data.extend([0xFF, 0xFF])  # multiple separators
    data.extend([1])  # key length
    data.extend(b'a')  # key
    data.extend([1])  # value length
    data.extend(b'1')  # value
    data.extend([0xFF, 0xFF, 0xFF])  # multiple separators
    data.extend([1])  # key length
    data.extend(b'b')  # key
    data.extend([1])  # value length
    data.extend(b'2')  # value
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    assert len(result) == 2
    assert result[0]['a'] == '1'
    assert result[1]['b'] == '2'

def test_decode_abr_empty_values():
    """Test ABR decoding with empty values."""
    data = bytearray()
    data.extend([3])  # key length
    data.extend(b'key')  # key
    data.extend([0])  # value length (empty)
    # no value bytes for empty string
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    assert len(result) == 1
    assert result[0]['key'] == ''

def test_decode_abr_multiple_kv_pairs():
    """Test ABR decoding with multiple key-value pairs in one record."""
    data = bytearray()
    data.extend([4])  # key length
    data.extend(b'name')  # key
    data.extend([4])  # value length
    data.extend(b'test')  # value
    data.extend([2])  # key length
    data.extend(b'id')  # key
    data.extend([1])  # value length
    data.extend(b'1')  # value
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    assert len(result) == 1
    assert result[0]['name'] == 'test'
    assert result[0]['id'] == '1'

def test_adversarial_none_input():
    """Test with None input to check type handling."""
    try:
        result = decode_abr_format(None)
        # Should either handle gracefully or raise exception
        assert isinstance(result, list)
        if len(result) == 1 and 'error' in result[0]:
            assert 'Failed to decode ABR format' in result[0]['error']
    except (TypeError, AttributeError):
        # Acceptable to raise exception for None input
        pass

def test_adversarial_malformed_length_prefix():
    """Test with length prefix that exceeds remaining data to trigger buffer overrun."""
    data = bytearray()
    data.extend([255])  # key length claims 255 bytes
    data.extend(b'short')  # but only 5 bytes available
    # This should cause pos + key_len > len(binary_data) condition
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    # Should handle gracefully without crashing
    assert isinstance(result, list)
    # Should either return empty list or error, but not crash
    assert len(result) == 0 or (len(result) == 1 and 'error' in result[0])

def test_adversarial_zero_length_keys():
    """Test with zero-length keys which might cause dictionary issues."""
    data = bytearray()
    data.extend([0])  # zero key length
    # no key bytes
    data.extend([5])  # value length
    data.extend(b'value')  # value
    data.extend([0xFF])  # separator
    data.extend([0])  # another zero key length
    data.extend([6])  # value length  
    data.extend(b'value2')  # value
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    assert isinstance(result, list)
    # Multiple empty keys should either overwrite each other or be handled specially
    if len(result) > 0 and 'error' not in result[0]:
        # If it succeeds, empty key should map to last value
        assert '' in result[0]

def test_adversarial_resource_exhaustion():
    """Test with data designed to cause excessive memory allocation."""
    data = bytearray()
    # Create a record that claims to have very long key/value but truncated data
    data.extend([200])  # key length 200
    data.extend(b'k' * 50)  # only 50 bytes of key data (truncated)
    # Missing value length and value - should hit boundary conditions
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    assert isinstance(result, list)
    # Should not crash or allocate excessive memory
    # Should handle truncated data gracefully
    assert len(result) <= 1

def test_adversarial_non_utf8_binary_data():
    """Test with binary data that contains invalid UTF-8 sequences."""
    data = bytearray()
    data.extend([4])  # key length
    data.extend([0xFF, 0xFE, 0xFD, 0xFC])  # invalid UTF-8 sequence as key
    data.extend([3])  # value length
    data.extend([0x80, 0x81, 0x82])  # invalid UTF-8 sequence as value
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    assert isinstance(result, list)
    # Should handle invalid UTF-8 with 'replace' error handling
    if len(result) == 1 and 'error' not in result[0]:
        # Should contain replacement characters or similar
        keys = list(result[0].keys())
        assert len(keys) == 1
        # The key and value should be decoded with replacement chars
        assert isinstance(keys[0], str)
        assert isinstance(result[0][keys[0]], str)

def test_strengthened_exact_boundary_key_length():
    """Test exact boundary condition where key length exactly equals remaining data."""
    data = bytearray()
    data.extend([5])  # key length of 5
    data.extend(b'exact')  # exactly 5 bytes, no room for value length byte
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    # Should return empty list since there's no room for value length
    # The boundary check should prevent reading beyond buffer
    assert isinstance(result, list)
    assert len(result) == 0

def test_strengthened_key_length_off_by_one():
    """Test key length that is exactly one byte too long."""
    data = bytearray()
    data.extend([6])  # key length claims 6 bytes
    data.extend(b'short')  # but only 5 bytes available (pos + 6 > len when len-pos = 5)
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    # Should handle the off-by-one boundary correctly
    assert isinstance(result, list)
    assert len(result) == 0

def test_strengthened_value_length_exact_boundary():
    """Test value length that exactly consumes remaining bytes."""
    data = bytearray()
    data.extend([3])  # key length
    data.extend(b'key')  # key
    data.extend([4])  # value length of 4
    data.extend(b'test')  # exactly 4 bytes, consuming all remaining data
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    # Should successfully parse since boundary is exact
    assert len(result) == 1
    assert result[0]['key'] == 'test'

def test_strengthened_value_length_boundary_violation():
    """Test value length that exceeds remaining data by exactly one byte."""
    data = bytearray()
    data.extend([3])  # key length
    data.extend(b'key')  # key  
    data.extend([5])  # value length claims 5 bytes
    data.extend(b'four')  # but only 4 bytes available
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    # Should detect boundary violation and not include partial record
    assert isinstance(result, list)
    assert len(result) == 0

def test_strengthened_position_arithmetic_precision():
    """Test precise position arithmetic with multiple boundary conditions."""
    data = bytearray()
    # First record - valid
    data.extend([1])  # key length
    data.extend(b'a')  # key
    data.extend([1])  # value length  
    data.extend(b'1')  # value
    data.extend([0xFF])  # separator
    # Second record - key length exactly equals remaining minus 1 for value length
    data.extend([2])  # key length of 2
    data.extend(b'bc')  # exactly 2 bytes
    # Missing value length byte - pos should equal len(binary_data) exactly
    
    encoded = base64.b64encode(data).decode('ascii')
    result = decode_abr_format(encoded)
    
    # Should parse first record but stop at second due to exact boundary
    assert len(result) == 1
    assert result[0]['a'] == '1'
    # Second record should not appear due to missing value length