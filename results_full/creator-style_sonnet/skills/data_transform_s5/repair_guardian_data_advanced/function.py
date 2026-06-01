def repair_guardian_data_advanced(base64_data):
    """
    Advanced GUARDIAN data repair tool that reconstructs corrupted text from backup blocks.

    Utility: Decodes base64 GUARDIAN data, identifies corruption patterns, extracts text 
             fragments from both primary and backup blocks, and attempts intelligent 
             reconstruction of the original message.

    Args:
        base64_data (str): Base64 encoded corrupted GUARDIAN data

    Returns:
        dict: Contains 'repaired_text', 'corrupted_blocks', 'success_status', and 'confidence'
    """
    import base64
    import struct
    import re

    try:
        # Decode base64 data
        raw_data = base64.b64decode(base64_data)

        # Parse header (first 16 bytes)
        if len(raw_data) < 16:
            return {
                'repaired_text': '',
                'corrupted_blocks': [],
                'success_status': False,
                'confidence': 0
            }

        header = raw_data[:16]
        magic = header[:8]
        block_count = struct.unpack('<Q', header[8:16])[0]

        # Detect corruption in block count (unrealistic values)
        if block_count > 1000000:
            block_count = (len(raw_data) - 16) // 32  # Estimate based on data size

        # Extract all text fragments and backup data
        text_fragments = []
        corrupted_block_ids = []
        backup_fragments = []

        offset = 16
        block_id = 0

        while offset < len(raw_data) - 4:
            try:
                # Try to read block header
                if offset + 4 <= len(raw_data):
                    block_type = struct.unpack('<I', raw_data[offset:offset+4])[0]
                    offset += 4

                    # Check for text block patterns
                    if block_type & 0xFF == 0x10:  # Text block indicator
                        # Extract text until null byte or control character
                        text_start = offset
                        text_data = b''

                        while offset < len(raw_data):
                            byte = raw_data[offset]
                            if byte == 0:  # Null terminator
                                break
                            elif 32 <= byte <= 126:  # Printable ASCII
                                text_data += bytes([byte])
                            elif byte in [0x1F, 0x0F]:  # Known corruption markers
                                corrupted_block_ids.append(block_id)
                                break
                            offset += 1

                        if text_data:
                            decoded_text = text_data.decode('ascii', errors='ignore')
                            if len(decoded_text) > 2:  # Filter out single chars
                                text_fragments.append(decoded_text)

                    # Look for backup/recovery patterns (0xFF markers)
                    elif block_type == 0xFFFFFFFF or (block_type >> 24) == 0xFF:
                        # Skip FF padding and look for backup text
                        while offset < len(raw_data) and raw_data[offset] in [0xFF, 0x00]:
                            offset += 1

                        # Try to extract backup text
                        backup_start = offset
                        backup_text = b''

                        while offset < len(raw_data) and offset - backup_start < 32:
                            byte = raw_data[offset]
                            if 32 <= byte <= 126:
                                backup_text += bytes([byte])
                            elif byte == 0:
                                break
                            offset += 1

                        if backup_text:
                            decoded_backup = backup_text.decode('ascii', errors='ignore')
                            if len(decoded_backup) > 2:
                                backup_fragments.append(decoded_backup)

                    else:
                        offset += min(32, len(raw_data) - offset)  # Skip unknown block

                block_id += 1

            except (struct.error, UnicodeDecodeError, IndexError):
                corrupted_block_ids.append(block_id)
                offset += 1
                block_id += 1

        # Combine primary and backup fragments
        all_fragments = text_fragments + backup_fragments

        # Intelligent reconstruction
        reconstructed_text = ""

        if all_fragments:
            # Remove duplicates while preserving order
            unique_fragments = []
            seen = set()
            for frag in all_fragments:
                if frag not in seen:
                    unique_fragments.append(frag)
                    seen.add(frag)

            # Sort by likely position in sentence
            sentence_starters = ['Another', 'This', 'The', 'A']
            sentence_enders = ['.', '!', '?']

            start_frags = [f for f in unique_fragments if any(f.startswith(s) for s in sentence_starters)]
            end_frags = [f for f in unique_fragments if any(f.endswith(e) for e in sentence_enders)]
            middle_frags = [f for f in unique_fragments if f not in start_frags and f not in end_frags]

            # Attempt reconstruction
            if start_frags:
                reconstructed_text = start_frags[0]
                remaining_frags = middle_frags + end_frags
            else:
                reconstructed_text = unique_fragments[0] if unique_fragments else ""
                remaining_frags = unique_fragments[1:]

            # Connect fragments intelligently
            for frag in remaining_frags:
                if reconstructed_text and not reconstructed_text.endswith(' '):
                    # Check if fragments can be concatenated
                    if reconstructed_text.endswith(frag[:1]):
                        # Overlap case - merge fragments
                        reconstructed_text += frag[1:]
                    else:
                        # Add space if needed
                        reconstructed_text += " " + frag
                else:
                    reconstructed_text += frag

        # Calculate success metrics
        success = len(all_fragments) > 0 and len(reconstructed_text) > 10
        confidence = min(100, (len(all_fragments) * 20) + (len(reconstructed_text) // 2))

        return {
            'repaired_text': reconstructed_text,
            'corrupted_blocks': corrupted_block_ids,
            'success_status': success,
            'confidence': confidence
        }

    except Exception as e:
        return {
            'repaired_text': f'Repair failed: {str(e)}',
            'corrupted_blocks': [],
            'success_status': False,
            'confidence': 0
        }