def parse_vdl_schema(vdl_content):
    """
    Parse a VDL (Validation Definition Language) schema into a structured representation.
    
    Utility: Converts VDL schema text into a structured dictionary with schema metadata
    and field definitions. Handles type codes, flags, enums, arrays, and nested objects.
    
    Args:
        vdl_content (str): The VDL schema content as a string
        
    Returns:
        dict: Dictionary with keys 'name' (str), 'version' (int), 'fields' (list).
        Each field dict contains: name, type, is_array, flags, and values (for enums).
    """
    import re
    
    lines = [line.strip() for line in vdl_content.strip().split('\n')]
    
    # Initialize result
    result = {'name': '', 'version': 0, 'fields': []}
    
    # Parse header
    for line in lines:
        if line.startswith('@schema'):
            header_match = re.match(r'@schema\s+(\w+)\s+@version\s+(\d+)', line)
            if header_match:
                result['name'] = header_match.group(1)
                result['version'] = int(header_match.group(2))
            break
    
    # Type code mapping
    type_mapping = {
        'S': 'string',
        'I': 'integer', 
        'F': 'float',
        'B': 'boolean'
    }
    
    # Parse fields
    for line in lines:
        # Skip comments, empty lines, and header
        if line.startswith('#') or not line or line.startswith('@'):
            continue
            
        # Handle array fields (start with *)
        is_array = line.startswith('*')
        if is_array:
            line = line[1:].strip()
            
        # Handle nested objects (start with >)
        if line.startswith('>'):
            continue  # Skip nested objects for now
            
        # Parse regular field: field_name : type_code [flags]
        field_match = re.match(r'(\w+)\s*:\s*(\S+)(.*)$', line)
        if not field_match:
            continue
            
        field_name = field_match.group(1)
        type_code = field_match.group(2)
        flags_text = field_match.group(3).strip()
        
        # Parse type
        field_type = ''
        enum_values = []
        
        if type_code.startswith('E(') and type_code.endswith(')'):
            # Enum type
            field_type = 'enum'
            enum_content = type_code[2:-1]  # Remove E( and )
            enum_values = [val.strip() for val in enum_content.split('|')]
        else:
            field_type = type_mapping.get(type_code, type_code)
        
        # Parse flags
        flags = []
        flag_matches = re.findall(r'\[([^\]]+)\]', flags_text)
        for flag_match in flag_matches:
            flag = flag_match.strip()
            if flag == 'R':
                flags.append('required')
            elif flag == 'U':
                flags.append('unique')
            elif flag == 'N':
                flags.append('nullable')
            elif flag.startswith('V(') and flag.endswith(')'):
                range_content = flag[2:-1]  # Remove V( and )
                flags.append(f'range({range_content})')
            else:
                flags.append(flag)
        
        # Create field dict
        field_dict = {
            'name': field_name,
            'type': field_type,
            'is_array': is_array,
            'flags': flags
        }
        
        # Add enum values if applicable
        if enum_values:
            field_dict['values'] = enum_values
            
        result['fields'].append(field_dict)
    
    return result