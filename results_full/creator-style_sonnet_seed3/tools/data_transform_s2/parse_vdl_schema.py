def parse_vdl_schema(schema_text):
    """
    Parse a VDL (Value Description Language) schema into a structured representation.
    
    Utility: Parses VDL schema text handling nested objects, array fields, enums, 
             constraints, and indentation-based structure into a Python dictionary.
    
    Args:
        schema_text (str): The VDL schema text to parse
        
    Returns:
        dict: Structured representation with schema metadata and parsed fields
    """
    import re
    
    lines = schema_text.strip().split('\n')
    result = {
        'schema_name': None,
        'version': None,
        'fields': {}
    }
    
    # Parse header
    header_match = re.match(r'@schema\s+(\w+)\s+@version\s+(\d+)', lines[0])
    if header_match:
        result['schema_name'] = header_match.group(1)
        result['version'] = int(header_match.group(2))
    
    def parse_constraints(constraint_text):
        constraints = []
        # Parse [R], [U], [V(...)], etc.
        constraint_matches = re.findall(r'\[([^\]]+)\]', constraint_text)
        for match in constraint_matches:
            if match == 'R':
                constraints.append({'type': 'required', 'value': True})
            elif match == 'U':
                constraints.append({'type': 'unique', 'value': True})
            elif match.startswith('V('):
                # Parse validation range like V(0..)
                range_match = re.match(r'V\((\d+|\*)\.\.(\d+|\*|)\)', match)
                if range_match:
                    min_val = None if range_match.group(1) == '*' else float(range_match.group(1))
                    max_val = None if range_match.group(2) in ('*', '') else float(range_match.group(2))
                    constraints.append({'type': 'validation', 'min': min_val, 'max': max_val})
        return constraints
    
    def parse_type(type_text):
        if type_text.startswith('E('):
            # Enum type
            enum_match = re.match(r'E\(([^)]+)\)', type_text)
            if enum_match:
                values = [v.strip() for v in enum_match.group(1).split('|')]
                return {'base_type': 'enum', 'values': values}
        
        type_map = {'S': 'string', 'F': 'float', 'I': 'integer', 'B': 'boolean'}
        return {'base_type': type_map.get(type_text, type_text)}
    
    def parse_fields(lines, start_idx, base_indent=0):
        fields = {}
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
                
            # Calculate indentation
            indent = len(line) - len(line.lstrip())
            if indent < base_indent:
                break
                
            if indent > base_indent:
                i += 1
                continue
                
            line_content = line.strip()
            
            # Check for nested object
            if line_content.startswith('> '):
                obj_match = re.match(r'>\s*(\w+)\s*:', line_content)
                if obj_match:
                    obj_name = obj_match.group(1)
                    nested_fields, next_i = parse_fields(lines, i + 1, indent + 4)
                    fields[obj_name] = {
                        'type': {'base_type': 'object'},
                        'is_array': False,
                        'constraints': [],
                        'fields': nested_fields
                    }
                    i = next_i
                    continue
            
            # Parse regular field
            field_match = re.match(r'^(\*\s*)?(\w+)\s*:\s*(.+)', line_content)
            if field_match:
                is_array = field_match.group(1) is not None
                field_name = field_match.group(2)
                field_def = field_match.group(3)
                
                # Split type and constraints
                type_part = field_def.split()[0]
                constraint_part = ' '.join(field_def.split()[1:])
                
                field_info = {
                    'type': parse_type(type_part),
                    'is_array': is_array,
                    'constraints': parse_constraints(constraint_part)
                }
                
                fields[field_name] = field_info
            
            i += 1
        
        return fields, i
    
    # Parse fields starting from line 1
    parsed_fields, _ = parse_fields(lines, 1, 0)
    result['fields'] = parsed_fields
    
    return result