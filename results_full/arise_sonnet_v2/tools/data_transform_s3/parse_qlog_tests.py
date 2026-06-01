import struct
import datetime

def test_parse_single_qlog_record():
    """Test parsing a single QLOG record."""
    # Create test data
    timestamp = 1640995200000000  # 2022-01-01 00:00:00 in microseconds
    flags = 0x01
    message = b"Test message"
    length = len(message)
    
    # Pack into QLOG format
    binary_data = struct.pack('<QBH', timestamp, flags, length) + message
    
    result = parse_qlog(binary_data)
    
    assert len(result) == 1
    record = result[0]
    assert 'error' not in record
    assert record['timestamp'] == timestamp
    assert record['flags'] == flags
    assert record['length'] == length
    assert record['message'] == "Test message"
    assert record['continuation'] == False

def test_parse_multiple_qlog_records():
    """Test parsing multiple QLOG records."""
    # Create two test records
    records_data = []
    
    # Record 1
    timestamp1 = 1640995200000000
    flags1 = 0x02
    message1 = b"First message"
    records_data.append(struct.pack('<QBH', timestamp1, flags1, len(message1)) + message1)
    
    # Record 2
    timestamp2 = 1640995260000000
    flags2 = 0x04  # continuation flag set
    message2 = b"Second message"
    records_data.append(struct.pack('<QBH', timestamp2, flags2, len(message2)) + message2)
    
    binary_data = b''.join(records_data)
    result = parse_qlog(binary_data)
    
    assert len(result) == 2
    
    # Check first record
    assert result[0]['timestamp'] == timestamp1
    assert result[0]['flags'] == flags1
    assert result[0]['message'] == "First message"
    assert result[0]['continuation'] == False
    
    # Check second record
    assert result[1]['timestamp'] == timestamp2
    assert result[1]['flags'] == flags2
    assert result[1]['message'] == "Second message"
    assert result[1]['continuation'] == True  # bit 2 is set

def test_parse_qlog_with_binary_message():
    """Test parsing QLOG with non-UTF8 binary message."""
    timestamp = 1640995200000000
    flags = 0x00
    message = b"\x00\x01\x02\x03\xff\xfe"
    length = len(message)
    
    binary_data = struct.pack('<QBH', timestamp, flags, length) + message
    result = parse_qlog(binary_data)
    
    assert len(result) == 1
    record = result[0]
    assert 'error' not in record
    # Should be hex encoded since it's not valid UTF-8
    assert record['message'] == "00010203fffe"

def test_parse_qlog_incomplete_data():
    """Test parsing incomplete QLOG data."""
    # Create incomplete header (only 5 bytes instead of 11)
    binary_data = b"\x00\x01\x02\x03\x04"
    
    result = parse_qlog(binary_data)
    
    # Should return empty list for incomplete header
    assert len(result) == 0

def test_parse_qlog_incomplete_message():
    """Test parsing QLOG with incomplete message."""
    timestamp = 1640995200000000
    flags = 0x00
    length = 20  # Claim 20 bytes
    message = b"short"  # But only provide 5 bytes
    
    binary_data = struct.pack('<QBH', timestamp, flags, length) + message
    result = parse_qlog(binary_data)
    
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'Incomplete message' in result[0]['error']

def test_parse_empty_qlog():
    """Test parsing empty QLOG data."""
    result = parse_qlog(b"")
    assert result == []

def test_parse_qlog_zero_length_message():
    """Test parsing QLOG record with zero-length message."""
    timestamp = 1640995200000000
    flags = 0x08
    length = 0
    
    binary_data = struct.pack('<QBH', timestamp, flags, length)
    result = parse_qlog(binary_data)
    
    assert len(result) == 1
    record = result[0]
    assert 'error' not in record
    assert record['message'] == ""
    assert record['length'] == 0

def test_adversarial_wrong_input_types():
    """Test parsing with wrong input types."""
    # Test with None
    result = parse_qlog(None)
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'Input must be bytes' in result[0]['error']
    
    # Test with string
    result = parse_qlog("not bytes")
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'Input must be bytes' in result[0]['error']
    
    # Test with int
    result = parse_qlog(123)
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'Input must be bytes' in result[0]['error']

def test_strengthened_exact_boundary_header():
    """Test exact boundary condition where we have exactly 11 bytes for header but no message."""
    # Create exactly 11 bytes - just enough for header
    timestamp = 1640995200000000
    flags = 0x01
    length = 5  # Claim we need 5 more bytes for message
    
    # Pack only the header (11 bytes) with no message bytes
    binary_data = struct.pack('<QBH', timestamp, flags, length)
    
    result = parse_qlog(binary_data)
    
    # Should detect incomplete message and return error
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'Incomplete message' in result[0]['error']

def test_strengthened_exact_boundary_complete_record():
    """Test exact boundary where we have exactly enough bytes for one complete record."""
    timestamp = 1640995200000000
    flags = 0x02
    message = b"exact"
    length = len(message)  # 5 bytes
    
    # Create exactly 11 + 5 = 16 bytes total
    binary_data = struct.pack('<QBH', timestamp, flags, length) + message
    
    result = parse_qlog(binary_data)
    
    # Should successfully parse exactly one record
    assert len(result) == 1
    assert 'error' not in result[0]
    assert result[0]['message'] == "exact"
    assert result[0]['length'] == 5

def test_strengthened_boundary_off_by_one_header():
    """Test boundary condition with exactly 10 bytes (one less than needed for header)."""
    # Create exactly 10 bytes - one byte short of complete header
    binary_data = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
    
    result = parse_qlog(binary_data)
    
    # Should return empty list since we can't even read a complete header
    assert len(result) == 0

def test_strengthened_boundary_off_by_one_message():
    """Test boundary condition with exactly one byte less than needed for message."""
    timestamp = 1640995200000000
    flags = 0x03
    length = 10  # Claim we need 10 bytes for message
    message = b"short123"  # But only provide 8 bytes (2 short)
    
    binary_data = struct.pack('<QBH', timestamp, flags, length) + message
    
    result = parse_qlog(binary_data)
    
    # Should detect incomplete message
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'Incomplete message' in result[0]['error']
    assert 'expected 10 bytes, only 8 available' in result[0]['error']

def test_strengthened_multiple_records_boundary():
    """Test boundary conditions with multiple records where second record is incomplete."""
    # First complete record
    timestamp1 = 1640995200000000
    flags1 = 0x01
    message1 = b"first"
    record1 = struct.pack('<QBH', timestamp1, flags1, len(message1)) + message1
    
    # Second record with incomplete header (only 8 bytes instead of 11)
    incomplete_header = b"\x00\x01\x02\x03\x04\x05\x06\x07"
    
    binary_data = record1 + incomplete_header
    result = parse_qlog(binary_data)
    
    # Should parse first record successfully and stop at incomplete second record
    assert len(result) == 1
    assert 'error' not in result[0]
    assert result[0]['message'] == "first"
    assert result[0]['timestamp'] == timestamp1