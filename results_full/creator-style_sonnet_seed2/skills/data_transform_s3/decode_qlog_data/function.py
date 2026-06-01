def decode_qlog_data(base64_data):
    """
    Decodes QLOG data from base64 format and analyzes its structure.
    
    Utility: Decodes base64 QLOG data and attempts to parse it as JSON or binary format,
             providing detailed information about the decoded content structure.
    
    Args:
        base64_data (str): Base64 encoded QLOG data string
    
    Returns:
        dict: Contains 'raw_bytes', 'text_content', 'json_records', 'hex_dump', 
              and 'analysis' of the decoded data
    """
    import base64
    import json
    
    try:
        # Decode base64 data
        decoded_bytes = base64.b64decode(base64_data)
        
        # Try to interpret as text
        text_content = ""
        try:
            text_content = decoded_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text_content = decoded_bytes.decode('utf-8', errors='replace')
        
        # Try to parse as JSON
        json_records = []
        if text_content.strip():
            # Try parsing entire content as JSON
            try:
                json_data = json.loads(text_content)
                if isinstance(json_data, list):
                    json_records = json_data
                else:
                    json_records = [json_data]
            except json.JSONDecodeError:
                # Try parsing line by line as NDJSON
                for line in text_content.strip().split('\n'):
                    if line.strip():
                        try:
                            record = json.loads(line.strip())
                            json_records.append(record)
                        except json.JSONDecodeError:
                            continue
        
        # Create hex dump for binary analysis
        hex_dump = ' '.join(f'{b:02x}' for b in decoded_bytes[:100])  # First 100 bytes
        if len(decoded_bytes) > 100:
            hex_dump += "..."
        
        # Analysis
        analysis = {
            'total_bytes': len(decoded_bytes),
            'is_text': all(32 <= b < 127 or b in [9, 10, 13] for b in decoded_bytes),
            'contains_json': len(json_records) > 0,
            'record_count': len(json_records),
            'first_bytes': list(decoded_bytes[:20]) if decoded_bytes else []
        }
        
        return {
            'raw_bytes': len(decoded_bytes),
            'text_content': text_content,
            'json_records': json_records,
            'hex_dump': hex_dump,
            'analysis': analysis
        }
        
    except Exception as e:
        return {
            'error': f'Failed to decode: {str(e)}',
            'raw_bytes': 0,
            'text_content': '',
            'json_records': [],
            'hex_dump': '',
            'analysis': {}
        }