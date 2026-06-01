def fetch_filtered_events() -> tuple[list[dict], list[dict]]:
    """Fetch error events and svc-1 events from the API, returning both result sets."""
    import urllib.request
    import urllib.parse
    import json
    import sys
    import traceback
    
    # Step 1: Set up the base API URL and prepare HTTP session with appropriate headers and timeout configuration
    base_url = "http://127.0.0.1:18080/api/events"
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Python/urllib'
    }
    timeout = 10
    
    error_events = []
    svc1_events = []
    
    # Step 2: Make first API request to fetch events filtered by type='error' with proper query parameters
    try:
        error_params = urllib.parse.urlencode({'type': 'error'})
        error_url = f"{base_url}?{error_params}"
        error_request = urllib.request.Request(error_url, headers=headers)
        
        with urllib.request.urlopen(error_request, timeout=timeout) as response:
            error_response_data = response.read().decode('utf-8')
            error_events = json.loads(error_response_data)
            
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        error_events = []
    
    # Step 3: Make second API request to fetch events filtered by source='svc-1' with proper query parameters
    try:
        svc1_params = urllib.parse.urlencode({'source': 'svc-1'})
        svc1_url = f"{base_url}?{svc1_params}"
        svc1_request = urllib.request.Request(svc1_url, headers=headers)
        
        with urllib.request.urlopen(svc1_request, timeout=timeout) as response:
            svc1_response_data = response.read().decode('utf-8')
            svc1_events = json.loads(svc1_response_data)
            
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        svc1_events = []
    
    # Step 4: Handle potential HTTP errors, network issues, and malformed JSON responses for both requests
    # (Already handled in the try-except blocks above)
    
    # Step 5: Parse and validate the response data structure from both API calls
    if not isinstance(error_events, list):
        error_events = []
    
    if not isinstance(svc1_events, list):
        svc1_events = []
    
    # Ensure all items in the lists are dictionaries
    error_events = [event for event in error_events if isinstance(event, dict)]
    svc1_events = [event for event in svc1_events if isinstance(event, dict)]
    
    # Step 6: Return tuple containing the two result sets, with empty lists as fallback for any failures
    return (error_events, svc1_events)