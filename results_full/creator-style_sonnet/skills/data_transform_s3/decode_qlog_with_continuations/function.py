def decode_qlog_with_continuations(base64_data):
    """
    Decode QLOG data and merge continuation entries with their parent entries.

    Utility: Decodes base64-encoded QLOG binary data, parses individual log entries,
             identifies continuation entries (flags bit 2 set), and merges them with
             preceding non-continuation entries using newline separators.

    Args:
        base64_data (str): Base64-encoded QLOG binary data

    Returns:
        list: List of merged log records as dictionaries with keys:
              timestamp, severity, subsystem, message
    """
    import base64
    import struct
    from datetime import datetime, timezone

    # Decode base64 data
    binary_data = base64.b64decode(base64_data)

    # Parse QLOG entries
    entries = []
    offset = 0

    while offset < len(binary_data):
        if offset + 6 > len(binary_data):
            break

        # Read header - let's try a different format
        # First, let's read the timestamp (4 bytes), flags (1 byte), subsystem (1 byte)
        timestamp_raw, flags, subsystem = struct.unpack('<IBB', binary_data[offset:offset + 6])
        offset += 6

        # Read message length (2 bytes)
        if offset + 2 > len(binary_data):
            break
        
        msg_len = struct.unpack('<H', binary_data[offset:offset + 2])[0]
        offset += 2

        # Read message
        if offset + msg_len > len(binary_data):
            break

        message = binary_data[offset:offset + msg_len].decode('utf-8', errors='ignore')
        offset += msg_len

        # Convert timestamp (assuming Unix timestamp)
        try:
            timestamp = datetime.fromtimestamp(timestamp_raw, tz=timezone.utc).isoformat()
        except (ValueError, OSError):
            # If timestamp is invalid, use a placeholder
            timestamp = f"raw_{timestamp_raw}"

        # Map severity (extract from lower bits of flags)
        severity_map = {0: "DEBUG", 1: "INFO", 2: "WARNING", 3: "ERROR", 4: "CRITICAL"}
        severity_bits = flags & 0x7
        severity = severity_map.get(severity_bits, f"LEVEL_{severity_bits}")

        # Check if this is a continuation entry (bit 2 set in flags)
        is_continuation = bool(flags & 0x4)

        entry = {
            "timestamp": timestamp,
            "severity": severity,
            "subsystem": subsystem,
            "message": message,
            "is_continuation": is_continuation,
            "flags": flags  # Keep for debugging
        }

        entries.append(entry)

        # Skip any padding bytes (0xFE pattern often used as separator)
        while offset < len(binary_data) and binary_data[offset] == 0xFE:
            offset += 1

    # Merge continuation entries with their parents
    merged_entries = []

    for entry in entries:
        if entry["is_continuation"] and merged_entries:
            # Append to the last entry (which should be the parent)
            merged_entries[-1]["message"] += "\n" + entry["message"]
        else:
            # Remove internal fields from the final output
            final_entry = {
                "timestamp": entry["timestamp"],
                "severity": entry["severity"],
                "subsystem": entry["subsystem"],
                "message": entry["message"]
            }
            merged_entries.append(final_entry)

    return merged_entries