def decode_qlog_data(base64_data):
    """
    Decode QLOG (Quantized Log Format) binary data into structured log records.

    Utility: Parses base64-encoded QLOG binary format containing timestamped log entries
    with severity levels, subsystem IDs, and message payloads separated by 0xFE 0xFE markers.

    Args:
        base64_data (str): Base64 encoded QLOG binary data

    Returns:
        list: List of dictionaries with keys:
            - timestamp (str): ISO format timestamp string
            - severity (str): Severity level name (TRACE, DEBUG, INFO, WARN, ERROR, FATAL)
            - subsystem (int): Subsystem ID (0-15)
            - message (str): UTF-8 decoded message text
            - flags (dict): Parsed flags (compressed, has_context, continuation)
    """
    import base64
    import struct
    from datetime import datetime, timezone, timedelta

    # Severity level mapping
    severity_names = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL']

    # Base epoch for timestamps (2025-01-01 00:00:00 UTC)
    base_epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # Decode base64 data
    binary_data = base64.b64decode(base64_data)

    # Split by separator 0xFE 0xFE
    separator = b'\xFE\xFE'
    raw_entries = binary_data.split(separator)

    parsed_entries = []

    for entry_data in raw_entries:
        if len(entry_data) < 8:  # Skip incomplete entries
            continue

        # Parse 8-byte header
        timestamp_raw = struct.unpack('>I', entry_data[0:4])[0]  # big-endian uint32
        packed_severity = entry_data[4]
        flags_byte = entry_data[5]
        payload_length = struct.unpack('>H', entry_data[6:8])[0]  # big-endian uint16

        # Extract severity and subsystem from packed byte
        severity_level = (packed_severity >> 4) & 0x0F
        subsystem_id = packed_severity & 0x0F

        # Parse flags
        flags = {
            'compressed': bool(flags_byte & 0x01),
            'has_context': bool(flags_byte & 0x02),
            'continuation': bool(flags_byte & 0x04)
        }

        # Extract payload (message)
        if len(entry_data) >= 8 + payload_length:
            message_bytes = entry_data[8:8+payload_length]
            message = message_bytes.decode('utf-8', errors='replace')
        else:
            message = ""

        # Convert timestamp to ISO format
        timestamp_dt = base_epoch + timedelta(seconds=timestamp_raw)
        timestamp_iso = timestamp_dt.isoformat()

        # Get severity name
        severity_name = severity_names[severity_level] if severity_level < len(severity_names) else f"UNKNOWN_{severity_level}"

        parsed_entries.append({
            'timestamp': timestamp_iso,
            'severity': severity_name,
            'subsystem': subsystem_id,
            'message': message,
            'flags': flags
        })

    return parsed_entries