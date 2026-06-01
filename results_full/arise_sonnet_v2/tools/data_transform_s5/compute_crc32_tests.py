import struct

def test_empty_data():
    """Test CRC32 of empty data"""
    result = compute_crc32(b'')
    # Empty data should have CRC32 of 0
    assert result == 0

def test_simple_data():
    """Test CRC32 of simple ASCII data"""
    data = b'hello'
    result = compute_crc32(data)
    # Should return a valid 32-bit unsigned integer
    assert isinstance(result, int)
    assert 0 <= result <= 0xFFFFFFFF

def test_different_data_different_crc():
    """Test that different data produces different CRC32"""
    data1 = b'hello'
    data2 = b'world'
    crc1 = compute_crc32(data1)
    crc2 = compute_crc32(data2)
    # Different data should produce different CRC32 (with very high probability)
    assert crc1 != crc2

def test_same_data_same_crc():
    """Test that same data produces same CRC32"""
    data = b'test data for consistency'
    crc1 = compute_crc32(data)
    crc2 = compute_crc32(data)
    assert crc1 == crc2

def test_binary_data():
    """Test CRC32 with binary data"""
    data = struct.pack('!IIHH', 0x12345678, 0xABCDEF00, 0x1234, 0x5678)
    result = compute_crc32(data)
    assert isinstance(result, int)
    assert 0 <= result <= 0xFFFFFFFF

def test_invalid_input():
    """Test CRC32 with invalid input type"""
    result = compute_crc32('not bytes')
    assert result == -1
    
    result = compute_crc32(123)
    assert result == -1
    
    result = compute_crc32(None)
    assert result == -1

def test_large_data():
    """Test CRC32 with larger data"""
    data = b'A' * 10000
    result = compute_crc32(data)
    assert isinstance(result, int)
    assert 0 <= result <= 0xFFFFFFFF

def test_corruption_detection():
    """Test that CRC32 can detect data corruption"""
    original_data = b'important data that must not be corrupted'
    original_crc = compute_crc32(original_data)
    
    # Simulate corruption by changing one byte
    corrupted_data = bytearray(original_data)
    corrupted_data[0] = (corrupted_data[0] + 1) % 256
    corrupted_crc = compute_crc32(bytes(corrupted_data))
    
    # CRC should detect the corruption
    assert original_crc != corrupted_crc

def test_independent_basic_crc32_computation():
    """Test basic CRC32 computation with known data"""
    # Test with empty data
    result = compute_crc32(b'')
    assert isinstance(result, int)
    assert result >= 0
    assert result <= 0xFFFFFFFF  # CRC32 is 32-bit unsigned
    
    # Test with simple data
    test_data = b'hello'
    result = compute_crc32(test_data)
    assert isinstance(result, int)
    assert result >= 0
    assert result <= 0xFFFFFFFF

def test_independent_crc32_deterministic():
    """Test that CRC32 computation is deterministic"""
    test_data = b'GUARDIAN_BLOCK_DATA'
    
    # Compute CRC32 multiple times for same data
    result1 = compute_crc32(test_data)
    result2 = compute_crc32(test_data)
    result3 = compute_crc32(test_data)
    
    # All results should be identical
    assert result1 == result2 == result3
    assert isinstance(result1, int)

def test_independent_crc32_different_data_different_results():
    """Test that different data produces different CRC32 values"""
    data1 = b'GUARDIAN_BLOCK_ORIGINAL'
    data2 = b'GUARDIAN_BLOCK_CORRUPTED'
    data3 = b'GUARDIAN_BLOCK_REPAIRED'
    
    crc1 = compute_crc32(data1)
    crc2 = compute_crc32(data2)
    crc3 = compute_crc32(data3)
    
    # Different data should produce different CRC values (highly likely)
    assert crc1 != crc2
    assert crc1 != crc3
    assert crc2 != crc3
    
    # All should be valid 32-bit unsigned integers
    for crc in [crc1, crc2, crc3]:
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFFFFFF

def test_independent_crc32_corruption_detection():
    """Test CRC32 can detect data corruption scenarios"""
    # Original block data
    original_data = b'GUARDIAN_BLOCK_' + b'\x00' * 16 + b'_END'
    original_crc = compute_crc32(original_data)
    
    # Simulate single bit corruption
    corrupted_data = bytearray(original_data)
    corrupted_data[10] ^= 0x01  # Flip one bit
    corrupted_crc = compute_crc32(bytes(corrupted_data))
    
    # CRC should detect corruption
    assert original_crc != corrupted_crc
    
    # Simulate byte corruption
    corrupted_data2 = bytearray(original_data)
    corrupted_data2[5] = 0xFF
    corrupted_crc2 = compute_crc32(bytes(corrupted_data2))
    
    assert original_crc != corrupted_crc2
    assert corrupted_crc != corrupted_crc2

def test_independent_crc32_input_validation():
    """Test CRC32 function handles various input types correctly"""
    # Test with bytes input (should work)
    valid_data = b'test_data_123'
    result = compute_crc32(valid_data)
    assert isinstance(result, int)
    
    # Test with different byte patterns
    binary_data = bytes([0x00, 0xFF, 0xAA, 0x55, 0x12, 0x34])
    result2 = compute_crc32(binary_data)
    assert isinstance(result2, int)
    assert 0 <= result2 <= 0xFFFFFFFF
    
    # Test with large data block (simulating GUARDIAN block)
    large_data = b'GUARDIAN' * 1000 + b'\x00' * 512
    result3 = compute_crc32(large_data)
    assert isinstance(result3, int)
    assert 0 <= result3 <= 0xFFFFFFFF
    
    # Verify different sizes produce valid results
    assert result != result2  # Different data should have different CRC
    assert result != result3

def test_adversarial_crc32_standard_library_comparison():
    """Test that our CRC32 implementation matches Python's standard library zlib.crc32"""
    import zlib
    
    test_cases = [
        b'',
        b'a',
        b'hello world',
        b'\x00\x01\x02\x03\xff\xfe\xfd\xfc',
        b'The quick brown fox jumps over the lazy dog',
        bytes(range(256))  # All possible byte values
    ]
    
    for data in test_cases:
        our_result = compute_crc32(data)
        # zlib.crc32 returns signed int, convert to unsigned 32-bit
        stdlib_result = zlib.crc32(data) & 0xFFFFFFFF
        assert our_result == stdlib_result, f"CRC32 mismatch for {data!r}: got {our_result}, expected {stdlib_result}"

def test_adversarial_memory_exhaustion_attack():
    """Test function behavior with extremely large input that could cause memory issues"""
    # Test with very large data that might cause memory allocation issues
    # Use a generator-like approach to avoid actually allocating huge memory in test
    large_size = 100 * 1024 * 1024  # 100MB
    
    # Create large data in chunks to test memory handling
    chunk_size = 1024 * 1024  # 1MB chunks
    large_data = b'A' * chunk_size
    
    # Test that function can handle reasonably large data without crashing
    result = compute_crc32(large_data)
    assert isinstance(result, int)
    assert 0 <= result <= 0xFFFFFFFF
    
    # Test with data containing all zero bytes (common edge case)
    zero_data = b'\x00' * 50000
    result_zeros = compute_crc32(zero_data)
    assert isinstance(result_zeros, int)
    assert 0 <= result_zeros <= 0xFFFFFFFF

def test_adversarial_unicode_and_encoding_confusion():
    """Test function with various string types that might cause encoding issues"""
    # Test with unicode strings that might be accidentally accepted
    unicode_string = "hello world"
    result = compute_crc32(unicode_string)
    assert result == -1, "Function should reject unicode strings"
    
    # Test with unicode containing non-ASCII characters
    unicode_non_ascii = "héllo wørld 🌍"
    result = compute_crc32(unicode_non_ascii)
    assert result == -1, "Function should reject unicode with non-ASCII"
    
    # Test with bytes-like objects that aren't actually bytes
    bytearray_data = bytearray(b'test')
    result = compute_crc32(bytearray_data)
    assert result == -1, "Function should reject bytearray (not bytes)"
    
    # Test with memoryview
    mv = memoryview(b'test')
    result = compute_crc32(mv)
    assert result == -1, "Function should reject memoryview"

def test_adversarial_integer_overflow_and_boundary():
    """Test potential integer overflow issues and boundary conditions"""
    # Test with data that might cause integer overflow in CRC computation
    # Pattern that exercises all bits in CRC calculation
    overflow_pattern = bytes([0xFF] * 1000)
    result = compute_crc32(overflow_pattern)
    assert isinstance(result, int)
    assert 0 <= result <= 0xFFFFFFFF, "Result must fit in 32-bit unsigned integer"
    
    # Test with alternating bit pattern
    alternating_pattern = bytes([0xAA, 0x55] * 1000)
    result2 = compute_crc32(alternating_pattern)
    assert isinstance(result2, int)
    assert 0 <= result2 <= 0xFFFFFFFF
    
    # Test single byte with maximum value
    max_byte = bytes([0xFF])
    result3 = compute_crc32(max_byte)
    assert isinstance(result3, int)
    assert 0 <= result3 <= 0xFFFFFFFF
    
    # Verify results are different (they should be with high probability)
    assert result != result2 != result3

def test_adversarial_exception_handling_bypass():
    """Test edge cases that might bypass exception handling"""
    # Test with objects that might pass isinstance check but fail later
    class FakeBytes:
        def __iter__(self):
            raise ValueError("Iteration failed")
        def __len__(self):
            return 5
    
    fake_bytes = FakeBytes()
    result = compute_crc32(fake_bytes)
    assert result == -1, "Function should handle iteration failures gracefully"
    
    # Test with bytes subclass that might have weird behavior
    class WeirdBytes(bytes):
        def __iter__(self):
            # Return non-integer values that might break the CRC calculation
            yield "not a byte"
            yield 256  # Out of byte range
            yield -1   # Negative value
    
    weird = WeirdBytes(b'abc')
    # This should either work normally (using bytes.__iter__) or return -1
    result = compute_crc32(weird)
    # Since WeirdBytes is a subclass of bytes, isinstance check passes
    # But the custom __iter__ might cause issues - function should handle gracefully
    assert result == -1 or (isinstance(result, int) and 0 <= result <= 0xFFFFFFFF)
    
    # Test with None masquerading as bytes through monkey patching
    result_none = compute_crc32(None)
    assert result_none == -1, "None should be rejected"