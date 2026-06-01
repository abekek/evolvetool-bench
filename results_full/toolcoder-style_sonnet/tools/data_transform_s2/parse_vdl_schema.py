def parse_vdl_schema(vdl_text: str) -> dict:
    """Parse a VDL schema with nested objects and array fields into a structured representation."""
    import re
    import traceback
    import sys
    
    try:
        # Step 1: Parse the schema header to extract schema name and version from @schema and @version directives
        lines = vdl_text.strip().split('\n')
        schema_name = None
        version = None
        field_lines = []
        
        for line in lines:
            line = line.rstrip()
            if line.startswith('@schema '):
                schema_name = line[8:].strip()
            elif line.startswith('@version '):
                version = line[9:].strip()
            elif line.strip() and not line.startswith('@'):
                field_lines.append(line)
        
        # Step 2: Split the schema into lines and process each line to identify field definitions, object nesting, and indentation levels
        processed_lines = []
        for line in field_lines:
            if line.strip():
                indent_level = len(line) - len(line.lstrip())
                content = line.strip()
                processed_lines.append((indent_level, content))
        
        # Step 3: Build a nested dictionary structure by tracking indentation depth and creating object hierarchies for '>' prefixed blocks
        def build_structure(lines_with_indent, start_idx=0, parent_indent=-1):
            structure = {}
            i = start_idx
            
            while i < len(lines_with_indent):
                indent, content = lines_with_indent[i]
                
                # If we've gone back to a lower indentation level, we're done with this block
                if indent <= parent_indent:
                    break
                
                # Handle nested objects (> prefix)
                if content.startswith('> ') and content.endswith(' :'):
                    obj_name = content[2:-2].strip()
                    # Find all lines that belong to this object (higher indentation)
                    nested_structure, next_i = build_structure(lines_with_indent, i + 1, indent)
                    structure[obj_name] = {
                        'type': 'object',
                        'fields': nested_structure
                    }
                    i = next_i
                    continue
                
                # Handle regular field definitions
                if ':' in content:
                    field_info = parse_field_definition(content)
                    if field_info:
                        field_name = field_info['name']
                        structure[field_name] = field_info
                
                i += 1
            
            return structure, i
        
        # Step 4: Parse each field line to extract field name, data type, constraints (R, U, V ranges), and special prefixes ('*' for arrays)
        def parse_field_definition(line):
            # Handle array prefix
            is_array = line.startswith('* ')
            if is_array:
                line = line[2:]
            
            # Split by colon to separate name from type and constraints
            if ':' not in line:
                return None
            
            name_part, type_part = line.split(':', 1)
            field_name = name_part.strip()
            type_and_constraints = type_part.strip()
            
            # Parse type and constraints
            field_type = None
            constraints = []
            
            # Split by spaces and brackets to identify components
            parts = re.findall(r'\S+|\[[^\]]*\]', type_and_constraints)
            
            for part in parts:
                if part.startswith('[') and part.endswith(']'):
                    # This is a constraint
                    constraint_content = part[1:-1]
                    constraints.append(parse_constraint(constraint_content))
                else:
                    # This should be the field type
                    if field_type is None:
                        field_type = part
            
            result = {
                'name': field_name,
                'type': field_type,
                'is_array': is_array,
                'constraints': constraints
            }
            
            return result
        
        # Step 5: Handle constraint parsing including enums E(value1|value2), validation ranges V(min..max), and open-ended ranges V(0..)
        def parse_constraint(constraint_str):
            constraint_str = constraint_str.strip()
            
            if constraint_str == 'R':
                return {'type': 'required'}
            elif constraint_str == 'U':
                return {'type': 'unique'}
            elif constraint_str.startswith('E(') and constraint_str.endswith(')'):
                enum_values = constraint_str[2:-1].split('|')
                return {'type': 'enum', 'values': enum_values}
            elif constraint_str.startswith('V(') and constraint_str.endswith(')'):
                range_str = constraint_str[2:-1]
                if '..' in range_str:
                    if range_str.endswith('..'):
                        # Open-ended range like "0.."
                        min_val = range_str[:-2]
                        return {'type': 'validation', 'min': float(min_val) if '.' in min_val else int(min_val), 'max': None}
                    else:
                        # Closed range like "0..100"
                        min_val, max_val = range_str.split('..')
                        return {
                            'type': 'validation',
                            'min': float(min_val) if '.' in min_val else int(min_val),
                            'max': float(max_val) if '.' in max_val else int(max_val)
                        }
                else:
                    # Single value
                    return {'type': 'validation', 'value': float(range_str) if '.' in range_str else int(range_str)}
            else:
                return {'type': 'unknown', 'value': constraint_str}
        
        # Step 6: Combine all parsed components into a final structured dictionary containing schema metadata and nested field definitions
        fields_structure, _ = build_structure(processed_lines)
        
        result = {
            'schema_name': schema_name,
            'version': version,
            'fields': fields_structure
        }
        
        return result
        
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        raise