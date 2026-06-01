import base64
import struct

def create_tpack_data(schema, records):
    """Helper to create valid TPACK data for testing"""
    data = bytearray()
    
    # Header: magic + record count
    data.extend(struct.pack('<II', 0x54504143, len(records)))
    
    # Schema: field count + fields
    data.extend(struct.pack('<I', len(schema)))
    for field_type, field_name in schema:
        data.append(field_type)
        data.extend(field_name.encode('utf-8'))
        data.append(0)  # null terminator
    
    # Records
    for record in records:
        for field_type, field_name in schema:
            value = record[field_name]
            if field_type == 0x01:  # int
                data.extend(struct.pack('<i', value))
            elif field_type == 0x02:  # double
                data.extend(struct.pack('<d', value))
            elif field_type == 0x03:  # bool
                data.append(1 if value else 0)
            elif field_type == 0x04:  # string
                utf8_bytes = value.encode('utf-8')
                data.extend(struct.pack('<I', len(utf8_bytes)))
                data.extend(utf8_bytes)
    
    return base64.b64encode(data).decode('ascii')

def test_deserialize_empty_records():
    """Test deserializing TPACK data with no records"""
    schema = [(0x01, 'id'), (0x04, 'name')]
    records = []
    tpack_data = create_tpack_data(schema, records)
    
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 0

def test_deserialize_single_record():
    """Test deserializing TPACK data with one record"""
    schema = [(0x01, 'id'), (0x04, 'name'), (0x03, 'active')]
    records = [{'id': 42, 'name': 'test', 'active': True}]
    tpack_data = create_tpack_data(schema, records)
    
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['id'] == 42
    assert result[0]['name'] == 'test'
    assert result[0]['active'] is True

def test_deserialize_multiple_records():
    """Test deserializing TPACK data with multiple records"""
    schema = [(0x01, 'id'), (0x02, 'score'), (0x03, 'available')]
    records = [
        {'id': 1, 'score': 95.5, 'available': True},
        {'id': 2, 'score': 87.2, 'available': False}
    ]
    tpack_data = create_tpack_data(schema, records)
    
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]['id'] == 1
    assert abs(result[0]['score'] - 95.5) < 1e-10
    assert result[0]['available'] is True
    assert result[1]['id'] == 2
    assert abs(result[1]['score'] - 87.2) < 1e-10
    assert result[1]['available'] is False

def test_deserialize_all_field_types():
    """Test deserializing TPACK data with all supported field types"""
    schema = [
        (0x01, 'int_field'),
        (0x02, 'double_field'), 
        (0x03, 'bool_field'),
        (0x04, 'string_field')
    ]
    records = [{
        'int_field': -123,
        'double_field': 3.14159,
        'bool_field': False,
        'string_field': 'hello world'
    }]
    tpack_data = create_tpack_data(schema, records)
    
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['int_field'] == -123
    assert abs(result[0]['double_field'] - 3.14159) < 1e-10
    assert result[0]['bool_field'] is False
    assert result[0]['string_field'] == 'hello world'

def test_deserialize_invalid_base64():
    """Test error handling for invalid base64 data"""
    result = deserialize_tpack('invalid base64!')
    assert isinstance(result, list)
    assert len(result) == 0

def test_deserialize_truncated_header():
    """Test error handling for truncated header"""
    short_data = base64.b64encode(b'short').decode('ascii')
    result = deserialize_tpack(short_data)
    assert isinstance(result, list)
    assert len(result) == 0

def test_deserialize_invalid_magic():
    """Test error handling for invalid magic number"""
    data = bytearray()
    data.extend(struct.pack('<II', 0x12345678, 0))  # Wrong magic
    data.extend(struct.pack('<I', 0))  # No fields
    
    tpack_data = base64.b64encode(data).decode('ascii')
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 0

def test_deserialize_empty_string_field():
    """Test deserializing record with empty string field"""
    schema = [(0x04, 'name')]
    records = [{'name': ''}]
    tpack_data = create_tpack_data(schema, records)
    
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['name'] == ''

def test_adversarial_huge_field_count():
    """Test with extremely large field count to trigger memory exhaustion"""
    data = bytearray()
    # Valid header
    data.extend(struct.pack('<II', 0x54504143, 0))  # 0 records
    # Huge field count that could cause memory issues
    data.extend(struct.pack('<I', 0xFFFFFFFF))  # Max uint32
    
    tpack_data = base64.b64encode(data).decode('ascii')
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 0

def test_adversarial_huge_string_length():
    """Test with malicious string length that exceeds available data"""
    data = bytearray()
    # Valid header with 1 record
    data.extend(struct.pack('<II', 0x54504143, 1))
    # Schema with 1 string field
    data.extend(struct.pack('<I', 1))
    data.append(0x04)  # String type
    data.extend(b'name\x00')  # Field name
    # Record with malicious string length
    data.extend(struct.pack('<I', 0xFFFFFFFF))  # Claim huge string length
    data.extend(b'short')  # But only provide short data
    
    tpack_data = base64.b64encode(data).decode('ascii')
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 0

def test_adversarial_unicode_field_names():
    """Test with malicious unicode sequences in field names"""
    data = bytearray()
    # Valid header with 1 record
    data.extend(struct.pack('<II', 0x54504143, 1))
    # Schema with field containing invalid UTF-8
    data.extend(struct.pack('<I', 1))
    data.append(0x01)  # Int type
    # Invalid UTF-8 sequence (incomplete multibyte)
    data.extend(b'\xc0\x80invalid\xff\xfe')
    data.append(0)  # null terminator
    # Valid record data
    data.extend(struct.pack('<i', 42))
    
    tpack_data = base64.b64encode(data).decode('ascii')
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 0

def test_adversarial_missing_null_terminator():
    """Test field name without null terminator causing buffer overrun"""
    data = bytearray()
    # Valid header with 1 record
    data.extend(struct.pack('<II', 0x54504143, 1))
    # Schema with field name missing null terminator
    data.extend(struct.pack('<I', 1))
    data.append(0x01)  # Int type
    data.extend(b'fieldname')  # No null terminator - should read to end of buffer
    
    tpack_data = base64.b64encode(data).decode('ascii')
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 0

def test_adversarial_record_count_mismatch():
    """Test when actual records don't match declared record count"""
    data = bytearray()
    # Header claims 1000 records but we provide data for 0
    data.extend(struct.pack('<II', 0x54504143, 1000))
    # Schema with 1 field
    data.extend(struct.pack('<I', 1))
    data.append(0x01)  # Int type
    data.extend(b'id\x00')  # Field name
    # No record data provided, but header claims 1000 records
    
    tpack_data = base64.b64encode(data).decode('ascii')
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 0

def test_strengthened_string_length_boundary():
    """Test exact boundary condition for string length validation"""
    data = bytearray()
    # Valid header with 1 record
    data.extend(struct.pack('<II', 0x54504143, 1))
    # Schema with 1 string field
    data.extend(struct.pack('<I', 1))
    data.append(0x04)  # String type
    data.extend(b'name\x00')  # Field name
    # String with length exactly equal to remaining bytes
    remaining_bytes = 5
    data.extend(struct.pack('<I', remaining_bytes))  # String length = remaining data
    data.extend(b'exact')  # Exactly 5 bytes
    
    tpack_data = base64.b64encode(data).decode('ascii')
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['name'] == 'exact'

def test_strengthened_string_offset_increment():
    """Test that string offset is properly incremented by string length"""
    schema = [(0x04, 'first'), (0x01, 'second')]
    records = [{'first': 'hello', 'second': 42}]
    tpack_data = create_tpack_data(schema, records)
    
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['first'] == 'hello'
    assert result[0]['second'] == 42
    
    # Test with different string lengths to ensure offset calculation is correct
    schema2 = [(0x04, 'short'), (0x04, 'longer_string'), (0x01, 'number')]
    records2 = [{'short': 'hi', 'longer_string': 'this is much longer', 'number': 123}]
    tpack_data2 = create_tpack_data(schema2, records2)
    
    result2 = deserialize_tpack(tpack_data2)
    assert isinstance(result, list)
    assert len(result2) == 1
    assert result2[0]['short'] == 'hi'
    assert result2[0]['longer_string'] == 'this is much longer'
    assert result2[0]['number'] == 123

def test_strengthened_multiple_strings_offset_tracking():
    """Test multiple consecutive strings to verify offset increments correctly"""
    schema = [(0x04, 'str1'), (0x04, 'str2'), (0x04, 'str3')]
    records = [{'str1': 'abc', 'str2': 'defgh', 'str3': 'ijklmnop'}]
    tpack_data = create_tpack_data(schema, records)
    
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['str1'] == 'abc'
    assert result[0]['str2'] == 'defgh' 
    assert result[0]['str3'] == 'ijklmnop'

def test_strengthened_string_boundary_off_by_one():
    """Test string length that is one byte too long for available data"""
    data = bytearray()
    # Valid header with 1 record
    data.extend(struct.pack('<II', 0x54504143, 1))
    # Schema with 1 string field
    data.extend(struct.pack('<I', 1))
    data.append(0x04)  # String type
    data.extend(b'name\x00')  # Field name
    # String length is 1 byte more than available data
    data.extend(struct.pack('<I', 6))  # Claim 6 bytes
    data.extend(b'short')  # Only provide 5 bytes
    
    tpack_data = base64.b64encode(data).decode('ascii')
    result = deserialize_tpack(tpack_data)
    assert isinstance(result, list)
    assert len(result) == 0

def test_strengthened_field_boundary_conditions():
    """Test boundary conditions for all field types at data end"""
    # Test integer field at exact boundary
    data = bytearray()
    data.extend(struct.pack('<II', 0x54504143, 1))  # 1 record
    data.extend(struct.pack('<I', 1))  # 1 field
    data.append(0x01)  # Int type
    data.extend(b'num\x00')  # Field name
    data.extend(struct.pack('<i', 42))  # Exactly 4 bytes for int
    
    tpack_data = base64.b64encode(data).decode('ascii')
    result = deserialize_tpack(tpack_data)
    assert len(result) == 1
    assert result[0]['num'] == 42
    
    # Test integer field missing 1 byte
    data2 = bytearray()
    data2.extend(struct.pack('<II', 0x54504143, 1))  # 1 record
    data2.extend(struct.pack('<I', 1))  # 1 field
    data2.append(0x01)  # Int type
    data2.extend(b'num\x00')  # Field name
    data2.extend(b'abc')  # Only 3 bytes instead of 4
    
    tpack_data2 = base64.b64encode(data2).decode('ascii')
    result2 = deserialize_tpack(tpack_data2)
    assert len(result2) == 0