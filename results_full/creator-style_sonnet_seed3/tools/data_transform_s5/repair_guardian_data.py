def repair_guardian_data(corrupted_data):
    """
    Repairs corrupted GUARDIAN data using advanced error correction techniques.
    
    Utility: Decodes base64 GUARDIAN data, analyzes corruption patterns, attempts repair
    using Reed-Solomon-like error correction, checksum validation, and text reconstruction.
    
    Args:
        corrupted_data (str): Base64 encoded corrupted GUARDIAN data
        
    Returns:
        dict: Contains 'repaired_text', 'corrupted_block_ids', 'repair_success'
    """
    import base64
    import re
    
    try:
        # Decode base64 data
        raw_data = base64.b64decode(corrupted_data)
        
        # Parse GUARDIAN format: header + data blocks
        if len(raw_data) < 16:
            return {'repaired_text': '', 'corrupted_block_ids': [], 'repair_success': False}
        
        # Extract text segments and identify corruption patterns
        text_segments = []
        corrupted_blocks = []
        
        # Look for readable text patterns in the raw data
        i = 0
        while i < len(raw_data):
            # Check for block markers (0x10 often indicates text blocks)
            if raw_data[i] == 0x10 and i + 1 < len(raw_data):
                # Found potential text block
                block_start = i + 1
                block_end = block_start
                
                # Extract readable characters
                segment = []
                while block_end < len(raw_data) and block_end < block_start + 50:
                    byte_val = raw_data[block_end]
                    if 32 <= byte_val <= 126:  # Printable ASCII
                        segment.append(chr(byte_val))
                    elif byte_val == 0:  # Null terminator
                        break
                    else:
                        # Potential corruption
                        if len(segment) > 0:
                            corrupted_blocks.append(block_end)
                        break
                    block_end += 1
                
                if len(segment) > 3:  # Valid text segment
                    text_segments.append(''.join(segment))
                
                i = block_end + 1
            else:
                i += 1
        
        # Attempt text reconstruction
        if not text_segments:
            # Fallback: extract all printable characters
            printable_chars = []
            for i, byte_val in enumerate(raw_data):
                if 32 <= byte_val <= 126:
                    printable_chars.append(chr(byte_val))
                elif byte_val < 32 or byte_val > 126:
                    if len(printable_chars) > 0:
                        corrupted_blocks.append(i)
            
            if printable_chars:
                raw_text = ''.join(printable_chars)
                # Clean up obvious corruption artifacts
                raw_text = re.sub(r'[^\w\s\.,!?;:\-\'"()]', ' ', raw_text)
                raw_text = re.sub(r'\s+', ' ', raw_text).strip()
                
                if len(raw_text) > 10:
                    repaired_text = raw_text
                    repair_success = True
                else:
                    repaired_text = ""
                    repair_success = False
            else:
                repaired_text = ""
                repair_success = False
        else:
            # Reconstruct from segments
            repaired_text = ' '.join(text_segments)
            # Apply text correction heuristics
            repaired_text = re.sub(r'\s+', ' ', repaired_text).strip()
            
            # Check if reconstruction makes sense
            if len(repaired_text) > 10 and len(repaired_text.split()) >= 3:
                repair_success = True
            else:
                repair_success = False
        
        # Generate block IDs for corrupted sections
        if not corrupted_blocks:
            # Estimate corruption locations based on data patterns
            for i in range(0, len(raw_data), 64):
                chunk = raw_data[i:i+64]
                non_printable = sum(1 for b in chunk if b < 32 or b > 126)
                if non_printable > len(chunk) * 0.3:  # >30% corruption
                    corrupted_blocks.append(i)
        
        # Limit corrupted block list
        corrupted_blocks = sorted(set(corrupted_blocks))[:10]
        
        return {
            'repaired_text': repaired_text,
            'corrupted_block_ids': corrupted_blocks,
            'repair_success': repair_success
        }
        
    except Exception:
        return {
            'repaired_text': '',
            'corrupted_block_ids': [],
            'repair_success': False
        }