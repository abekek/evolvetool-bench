def get_authentication_scheme_info(api_url: str = "http://127.0.0.1:18080/auth/info") -> dict:
    """Retrieves and analyzes authentication scheme information from the specified API endpoint."""
    import urllib.request
    import urllib.error
    import json
    import sys
    import traceback
    
    try:
        # Step 1: Make HTTP GET request to the auth info endpoint with proper headers handling
        request = urllib.request.Request(api_url)
        request.add_header('User-Agent', 'Python/urllib')
        request.add_header('Accept', 'application/json')
        
        with urllib.request.urlopen(request, timeout=10) as response:
            response_data = response.read().decode('utf-8')
            status_code = response.getcode()
        
        # Step 2: Parse the JSON response to extract authentication scheme details
        try:
            auth_info = json.loads(response_data)
        except json.JSONDecodeError:
            # If not JSON, treat as plain text response
            auth_info = {"raw_response": response_data}
        
        # Step 3: Analyze the response structure to identify the authentication method (Bearer, Basic, API Key, etc.)
        auth_scheme = "Unknown"
        scheme_details = {}
        
        if isinstance(auth_info, dict):
            # Common fields that indicate auth schemes
            scheme_indicators = {
                'bearer': ['bearer', 'jwt', 'token', 'access_token'],
                'basic': ['basic', 'username', 'password', 'credentials'],
                'api_key': ['api_key', 'apikey', 'key', 'x-api-key'],
                'oauth': ['oauth', 'client_id', 'client_secret', 'authorization_code'],
                'digest': ['digest', 'realm', 'nonce']
            }
            
            response_str = json.dumps(auth_info).lower()
            detected_schemes = []
            
            for scheme, indicators in scheme_indicators.items():
                if any(indicator in response_str for indicator in indicators):
                    detected_schemes.append(scheme)
            
            if detected_schemes:
                auth_scheme = detected_schemes[0].title()  # Take first match
                if len(detected_schemes) > 1:
                    scheme_details['multiple_schemes_detected'] = detected_schemes
        
        # Step 4: Extract relevant authentication parameters and requirements from the response
        auth_parameters = {}
        if isinstance(auth_info, dict):
            # Extract common auth-related fields
            for key, value in auth_info.items():
                key_lower = key.lower()
                if any(term in key_lower for term in ['auth', 'token', 'key', 'scheme', 'method', 'type', 'bearer', 'basic']):
                    auth_parameters[key] = value
        
        # Step 5: Format and return a structured summary of the authentication scheme and its details
        result = {
            'api_url': api_url,
            'status_code': status_code,
            'authentication_scheme': auth_scheme,
            'raw_response': auth_info,
            'authentication_parameters': auth_parameters,
            'scheme_details': scheme_details,
            'analysis_summary': f"API uses {auth_scheme} authentication scheme" if auth_scheme != "Unknown" else "Could not determine authentication scheme from response"
        }
        
        return result
        
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='ignore') if hasattr(e, 'read') else str(e)
        print(f"HTTP Error: {e.code} - {e.reason}", file=sys.stderr)
        print(f"Error body: {error_body}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return {
            'api_url': api_url,
            'error': f"HTTP {e.code}: {e.reason}",
            'error_body': error_body,
            'authentication_scheme': 'Error - Could not retrieve'
        }
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return {
            'api_url': api_url,
            'error': f"Connection error: {e.reason}",
            'authentication_scheme': 'Error - Could not connect'
        }
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return {
            'api_url': api_url,
            'error': f"Unexpected error: {str(e)}",
            'authentication_scheme': 'Error - Unexpected failure'
        }