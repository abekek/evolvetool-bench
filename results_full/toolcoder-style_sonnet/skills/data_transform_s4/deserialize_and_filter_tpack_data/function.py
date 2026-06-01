def deserialize_and_filter_tpack_data(encoded_data: str) -> list[dict]:
    """Deserialize TPACK data and filter records where 'available' is True."""
    import base64
    import struct
    import sys
    import traceback
    
    try:
        # Step 1: Decode the base64-encoded TPACK data to get raw bytes
        raw_bytes = base64.b64decode(encoded_data)
        
        # Step 2: Parse the binary TPACK format to extract individual records and their fields
        records = []
        offset = 0
        
        while offset < len(raw_bytes):
            record = {}
            
            # Parse each field in the record
            while offset < len(raw_bytes):
                # Check if we've hit a record boundary (indicated by field type 0x03 for next record)
                if offset + 1 < len(raw_bytes) and raw_bytes[offset] == 0x03:
                    # Field type 0x03 indicates start of next record or end
                    field_type = raw_bytes[offset]
                    offset += 1
                    
                    # Read field name length and name
                    name_length = raw_bytes[offset]
                    offset += 1
                    field_name = raw_bytes[offset:offset + name_length].decode('utf-8')
                    offset += name_length
                    
                    if field_name == 'sku':
                        # This indicates start of a new record
                        if record:  # If we have a current record, save it
                            records.append(record)
                            record = {}
                        
                        # Read the sku value
                        value_length = raw_bytes[offset]
                        offset += 1
                        record[field_name] = raw_bytes[offset:offset + value_length].decode('utf-8')
                        offset += value_length
                        break
                    else:
                        # Read field value based on type
                        if field_name in ['available']:
                            # Boolean field - no additional length byte
                            record[field_name] = True  # Type 0x03 seems to indicate True
                        else:
                            # Other field types
                            value_length = raw_bytes[offset]
                            offset += 1
                            record[field_name] = raw_bytes[offset:offset + value_length].decode('utf-8')
                            offset += value_length
                        break
                
                elif raw_bytes[offset] == 0x04:  # String field
                    field_type = raw_bytes[offset]
                    offset += 1
                    
                    # Read field name
                    name_length = raw_bytes[offset]
                    offset += 1
                    field_name = raw_bytes[offset:offset + name_length].decode('utf-8')
                    offset += name_length
                    
                    # Read field value
                    value_length = raw_bytes[offset]
                    offset += 1
                    value = raw_bytes[offset:offset + value_length].decode('utf-8')
                    offset += value_length
                    
                    record[field_name] = value
                
                elif raw_bytes[offset] == 0x13:  # Double/float field
                    field_type = raw_bytes[offset]
                    offset += 1
                    
                    # Read field name
                    name_length = raw_bytes[offset]
                    offset += 1
                    field_name = raw_bytes[offset:offset + name_length].decode('utf-8')
                    offset += name_length
                    
                    # Read 8-byte double value
                    value = struct.unpack('>d', raw_bytes[offset:offset + 8])[0]
                    offset += 8
                    
                    record[field_name] = value
                
                elif raw_bytes[offset] == 0x10:  # Integer field
                    field_type = raw_bytes[offset]
                    offset += 1
                    
                    # Read field name
                    name_length = raw_bytes[offset]
                    offset += 1
                    field_name = raw_bytes[offset:offset + name_length].decode('utf-8')
                    offset += name_length
                    
                    # Read integer value (appears to be single byte for qty)
                    if field_name == 'qty':
                        value = raw_bytes[offset]
                        offset += 1
                    else:
                        # For other integers, might be 4 bytes
                        value = struct.unpack('>I', raw_bytes[offset:offset + 4])[0]
                        offset += 4
                    
                    record[field_name] = value
                
                elif raw_bytes[offset] == 0x02:  # Boolean false
                    field_type = raw_bytes[offset]
                    offset += 1
                    record['available'] = False
                
                else:
                    offset += 1
        
        # Add the last record if it exists
        if record:
            records.append(record)
        
        # Step 3: Convert each parsed record into a dictionary with appropriate field names and data types
        # (Already done above during parsing)
        
        # Step 4: Filter the deserialized records to only include those where the 'available' field is True
        filtered_records = [record for record in records if record.get('available', False) == True]
        
        # Step 5: Return the filtered list of records as dictionaries
        return filtered_records
        
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        return []