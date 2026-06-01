def parse_vdl_schema(schema_text):
    """
    Parse a VDL (Versatile Data Language) schema with nested objects and array fields.
    
    Utility: Converts VDL schema text into a structured dictionary representation,
    handling nested objects (> prefix), array fields (* prefix), data types, and constraints.
    
    Args:
        schema_text (str): The VDL schema text to parse
        
    Returns:
        dict: Structured representation containing schema metadata and field definitions
    """
    import re
    
    lines = schema_text.strip().split('\n')
    result = {'fields': {}}
    
    # Parse schema header
    header_match = re.match(r'@schema\s+(\w+)\s+@version\s+(\d+)', lines[0])
    if header_match:
        result['schema_name'] = header_match.group(1)
        result['version'] = int(header_match.group(2))
    
    def parse_constraints(constraint_str):
        constraints = []
        if not constraint_str:
            return constraints
            
        # Find all constraint patterns
        constraint_patterns = re.findall(r'\[([^\]]+)\]', constraint_str)
        for pattern in constraint_patterns:
            if pattern == 'R':
                constraints.append({'type': 'required', 'value': True})
            elif pattern == 'U':
                constraints.append({'type': 'unique', 'value': True})
            elif pattern.startswith('V('):
                range_match = re.match(r'V\((\d+)\.\.(\d*)\)', pattern)
                if range_match:
                    min_val = int(range_match.group(1))
                    max_val = range_match.group(2)
                    max_val = int(max_val) if max_val else None
                    constraints.append({'type': 'validation', 'min': min_val, 'max': max_val})
        return constraints
    
    def parse_field_line(line):
        # Remove leading whitespace for processing but keep track of indentation
        indent_level = (len(line) - len(line.lstrip())) // 4
        line = line.strip()
        
        # Check for array prefix
        is_array = line.startswith('*')
        if is_array:
            line = line[1:].strip()
        
        # Check for nested object prefix
        is_nested = line.startswith('>')
        if is_nested:
            line = line[1:].strip()
        
        # Split field definition
        if ':' not in line:
            return None
            
        field_name, rest = line.split(':', 1)
        field_name = field_name.strip()
        rest = rest.strip()
        
        # Parse data type and constraints
        data_type = None
        constraints_str = rest
        
        # Handle enum types E(option1|option2|...)
        enum_match = re.match(r'E\(([^)]+)\)', rest)
        if enum_match:
            data_type = 'enum'
            enum_values = [val.strip() for val in enum_match.group(1).split('|')]
            constraints_str = rest[enum_match.end():].strip()
        else:
            # Handle basic types
            type_match = re.match(r'([SFIN])', rest)
            if type_match:
                type_map = {'S': 'string', 'F': 'float', 'I': 'integer', 'N': 'number'}
                data_type = type_map.get(type_match.group(1), 'string')
                constraints_str = rest[1:].strip()
        
        field_def = {
            'name': field_name,
            'type': data_type,
            'is_array': is_array,
            'is_nested': is_nested,
            'indent_level': indent_level,
            'constraints': parse_constraints(constraints_str)
        }
        
        if enum_match:
            field_def['enum_values'] = enum_values
            
        return field_def
    
    def build_nested_structure(fields):
        stack = [{}]
        level_stack = [-1]
        
        for field in fields:
            current_level = field['indent_level']
            
            # Pop stack until we're at the right level
            while len(level_stack) > 1 and level_stack[-1] >= current_level:
                stack.pop()
                level_stack.pop()
            
            # Prepare field data
            field_data = {
                'type': field['type'],
                'constraints': field['constraints']
            }
            
            if field.get('enum_values'):
                field_data['enum_values'] = field['enum_values']
            
            if field['is_array']:
                field_data['is_array'] = True
                
            if field['is_nested']:
                field_data['fields'] = {}
                stack.append(field_data['fields'])
                level_stack.append(current_level)
            
            # Add field to current context
            stack[-1][field['name']] = field_data
        
        return stack[0]
    
    # Parse all field lines
    parsed_fields = []
    for line in lines[1:]:
        if line.strip():
            field = parse_field_line(line)
            if field:
                parsed_fields.append(field)
    
    # Build nested structure
    result['fields'] = build_nested_structure(parsed_fields)
    
    return result