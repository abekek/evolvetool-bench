def deserialize_tpack(base64_data):
    """
    Deserialize TPACK (Tagged Pack Format) binary data into Python objects.

    Utility:
    Parses base64-encoded TPACK binary format with type tags, varints, and nested structures.
    Supports null, boolean, numeric, string, array, and map types according to TPACK specification.

    Args:
    base64_data (str): Base64 encoded TPACK binary data string

    Returns:
    object: Deserialized Python data structure (dict, list, str, int, float, bool, or None)
    """
    import base64
    import struct

    def parse_varint(data, offset):
        result = 0
        shift = 0
        while offset < len(data):
            byte = data[offset]
            offset += 1
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
        return result, offset

    def parse_string(data, offset):
        """Parse a string with varint length prefix"""
        length, offset = parse_varint(data, offset)
        value = data[offset:offset+length].decode('utf-8')
        return value, offset + length

    def parse_value(data, offset):
        if offset >= len(data):
            raise ValueError("Unexpected end of data")

        tag = data[offset]
        offset += 1

        if tag == 0x01:  # null
            return None, offset
        elif tag == 0x02:  # false
            return False, offset
        elif tag == 0x03:  # true
            return True, offset
        elif tag == 0x10:  # uint8
            return data[offset], offset + 1
        elif tag == 0x11:  # uint16 big-endian
            value = struct.unpack('>H', data[offset:offset+2])[0]
            return value, offset + 2
        elif tag == 0x12:  # int32 big-endian
            value = struct.unpack('>i', data[offset:offset+4])[0]
            return value, offset + 4
        elif tag == 0x13:  # float64 big-endian
            value = struct.unpack('>d', data[offset:offset+8])[0]
            return value, offset + 8
        elif tag == 0x20:  # string
            return parse_string(data, offset)
        elif tag == 0x30:  # array
            count, offset = parse_varint(data, offset)
            array = []
            for _ in range(count):
                element, offset = parse_value(data, offset)
                array.append(element)
            return array, offset
        elif tag == 0x40:  # map
            count, offset = parse_varint(data, offset)
            result = {}
            for _ in range(count):
                # Parse key - keys are always strings but without the 0x20 tag in maps
                key, offset = parse_string(data, offset)
                # Parse value
                value, offset = parse_value(data, offset)
                result[key] = value
            return result, offset
        else:
            raise ValueError(f"Unknown tag: 0x{tag:02x} at offset {offset-1}")

    # Decode base64 data
    binary_data = base64.b64decode(base64_data)

    # Parse the data
    result, _ = parse_value(binary_data, 0)
    return result