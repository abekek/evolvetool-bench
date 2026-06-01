import hmac
import hashlib

def test_hmac_sha256_basic():
    """Test basic HMAC-SHA256 generation with known inputs."""
    secret = "my_secret_key"
    message = "hello world"
    
    result = hmac_sha256(secret, message)
    
    # Verify it's a valid hex string of correct length (SHA-256 = 64 hex chars)
    assert len(result) == 64
    assert all(c in '0123456789abcdef' for c in result)
    
    # Verify it matches standard library implementation
    expected = hmac.new(secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()
    assert result == expected

def test_hmac_sha256_empty_inputs():
    """Test HMAC-SHA256 with empty secret and message."""
    result = hmac_sha256("", "")
    
    # Should still produce valid hash
    assert len(result) == 64
    assert all(c in '0123456789abcdef' for c in result)
    
    # Verify against standard library
    expected = hmac.new(b"", b"", hashlib.sha256).hexdigest()
    assert result == expected

def test_hmac_sha256_special_characters():
    """Test HMAC-SHA256 with special characters and unicode."""
    secret = "key_with_!@#$%^&*()"
    message = "message with unicode: 你好世界"
    
    result = hmac_sha256(secret, message)
    
    # Verify format
    assert len(result) == 64
    assert all(c in '0123456789abcdef' for c in result)
    
    # Verify against standard library
    expected = hmac.new(secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()
    assert result == expected

def test_hmac_sha256_consistency():
    """Test that same inputs always produce same output."""
    secret = "consistent_key"
    message = "consistent_message"
    
    result1 = hmac_sha256(secret, message)
    result2 = hmac_sha256(secret, message)
    
    assert result1 == result2
    assert len(result1) == 64

def test_hmac_sha256_different_keys():
    """Test that different keys produce different hashes."""
    message = "same_message"
    
    result1 = hmac_sha256("key1", message)
    result2 = hmac_sha256("key2", message)
    
    assert result1 != result2
    assert len(result1) == 64
    assert len(result2) == 64

def test_independent_basic_hmac_sha256_functionality():
    """Test basic HMAC-SHA256 generation with simple inputs"""
    secret = "test_secret"
    message = "test_message"
    
    result = hmac_sha256(secret, message)
    
    # Verify it returns a string
    assert isinstance(result, str)
    
    # Compute expected HMAC-SHA256 using standard library
    expected = hmac.new(
        secret.encode('utf-8'), 
        message.encode('utf-8'), 
        hashlib.sha256
    ).hexdigest()
    
    assert result == expected

def test_independent_empty_inputs():
    """Test HMAC-SHA256 with empty secret and message"""
    # Test empty message
    result1 = hmac_sha256("secret", "")
    expected1 = hmac.new(
        "secret".encode('utf-8'),
        "".encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    assert result1 == expected1
    
    # Test empty secret
    result2 = hmac_sha256("", "message")
    expected2 = hmac.new(
        "".encode('utf-8'),
        "message".encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    assert result2 == expected2
    
    # Test both empty
    result3 = hmac_sha256("", "")
    expected3 = hmac.new(
        "".encode('utf-8'),
        "".encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    assert result3 == expected3

def test_independent_api_authentication_format():
    """Test HMAC-SHA256 with realistic API authentication data"""
    # Simulate typical API auth scenario with timestamp and request data
    api_secret = "sk_test_1234567890abcdef"
    timestamp = "1640995200"
    method = "GET"
    path = "/api/users"
    
    # Common API auth message format: timestamp + method + path
    message = timestamp + method + path
    
    result = hmac_sha256(api_secret, message)
    expected = hmac.new(
        api_secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    assert result == expected
    assert len(result) == 64  # SHA256 hex digest is always 64 characters
    assert all(c in '0123456789abcdef' for c in result)  # Valid hex

def test_independent_special_characters_and_unicode():
    """Test HMAC-SHA256 with special characters and unicode content"""
    secret = "secret!@#$%^&*()_+-={}[]|\\:;\"'<>?,./ "
    message = "Hello 世界! Special chars: ñáéíóú €£¥"
    
    result = hmac_sha256(secret, message)
    expected = hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    assert result == expected
    assert isinstance(result, str)
    assert len(result) == 64

def test_independent_deterministic_output():
    """Test that HMAC-SHA256 produces consistent, deterministic output"""
    secret = "consistent_secret_key"
    message = "consistent_message_content"
    
    # Generate hash multiple times
    result1 = hmac_sha256(secret, message)
    result2 = hmac_sha256(secret, message)
    result3 = hmac_sha256(secret, message)
    
    # All results should be identical
    assert result1 == result2 == result3
    
    # Verify against expected value
    expected = hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    assert result1 == expected
    
    # Different inputs should produce different outputs
    different_result = hmac_sha256(secret, message + "x")
    assert different_result != result1

def test_adversarial_none_inputs():
    """Test HMAC-SHA256 with None inputs"""
    # Test None secret
    result1 = hmac_sha256(None, "message")
    assert result1.startswith("Error generating HMAC-SHA256:")
    
    # Test None message
    result2 = hmac_sha256("secret", None)
    assert result2.startswith("Error generating HMAC-SHA256:")
    
    # Test both None
    result3 = hmac_sha256(None, None)
    assert result3.startswith("Error generating HMAC-SHA256:")

def test_bytes_input():
    """Test HMAC-SHA256 with bytes inputs"""
    secret_bytes = b"secret_key"
    message_bytes = b"test_message"
    
    result = hmac_sha256(secret_bytes, message_bytes)
    expected = hmac.new(secret_bytes, message_bytes, hashlib.sha256).hexdigest()
    
    assert result == expected
    assert len(result) == 64
    assert all(c in '0123456789abcdef' for c in result)

def test_invalid_input_types():
    """Test HMAC-SHA256 with invalid input types"""
    # Test with integer inputs
    result1 = hmac_sha256(123, "message")
    assert result1.startswith("Error generating HMAC-SHA256:")
    
    result2 = hmac_sha256("secret", 456)
    assert result2.startswith("Error generating HMAC-SHA256:")

def test_strengthened_none_secret_only():
    """Test that None secret alone triggers error, not just when both are None"""
    result = hmac_sha256(None, "valid_message")
    assert "Error generating HMAC-SHA256: Input cannot be None" in result
    assert not result.startswith("Error generating HMAC-SHA256: Invalid secret type")
    assert len(result) != 64  # Should not be a valid hash

def test_strengthened_none_message_only():
    """Test that None message alone triggers error, not just when both are None"""
    result = hmac_sha256("valid_secret", None)
    assert "Error generating HMAC-SHA256: Input cannot be None" in result
    assert not result.startswith("Error generating HMAC-SHA256: Invalid message type")
    assert len(result) != 64  # Should not be a valid hash

def test_strengthened_none_check_independence():
    """Test that None check works independently for each parameter"""
    # If 'or' was changed to 'and', only both being None would trigger error
    # Test each None case produces the specific None error message
    result1 = hmac_sha256(None, "message")
    result2 = hmac_sha256("secret", None)
    
    # Both should produce the None error, not type errors
    assert result1 == "Error generating HMAC-SHA256: Input cannot be None"
    assert result2 == "Error generating HMAC-SHA256: Input cannot be None"
    
    # Verify these are not valid hashes
    assert not all(c in '0123456789abcdef' for c in result1)
    assert not all(c in '0123456789abcdef' for c in result2)

def test_strengthened_none_vs_invalid_type():
    """Test that None inputs are caught before type checking"""
    # None should trigger None error, not type error
    result_none_secret = hmac_sha256(None, "test")
    result_none_message = hmac_sha256("test", None)
    
    # Should get None error, not type error
    assert result_none_secret == "Error generating HMAC-SHA256: Input cannot be None"
    assert result_none_message == "Error generating HMAC-SHA256: Input cannot be None"
    
    # Compare with actual type errors
    result_int_secret = hmac_sha256(123, "test")
    result_int_message = hmac_sha256("test", 123)
    
    assert "Invalid secret type" in result_int_secret
    assert "Invalid message type" in result_int_message
    
    # None errors should be different from type errors
    assert result_none_secret != result_int_secret
    assert result_none_message != result_int_message

def test_strengthened_mixed_none_and_valid():
    """Test mixed None and valid inputs to ensure proper error precedence"""
    # Test that when one input is None and other is valid, None error is returned
    # This would fail if 'or' was changed to 'and' in the None check
    
    valid_secret = "test_secret"
    valid_message = "test_message"
    
    # One None, one valid - should still error
    result1 = hmac_sha256(None, valid_message)
    result2 = hmac_sha256(valid_secret, None)
    
    # Both should return None error
    expected_error = "Error generating HMAC-SHA256: Input cannot be None"
    assert result1 == expected_error
    assert result2 == expected_error
    
    # Verify that valid inputs work
    result_valid = hmac_sha256(valid_secret, valid_message)
    assert len(result_valid) == 64
    assert all(c in '0123456789abcdef' for c in result_valid)
    
    # None cases should not produce valid hashes
    assert result1 != result_valid
    assert result2 != result_valid