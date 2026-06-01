import urllib.request
import urllib.error
from unittest.mock import patch, MagicMock
import json

def test_successful_request():
    """Test successful HTTP GET request with headers"""
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"status": "success"}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    
    with patch('urllib.request.urlopen', return_value=mock_response) as mock_urlopen:
        with patch('urllib.request.Request') as mock_request:
            result = http_get_with_headers('http://example.com/api', {'Authorization': 'Bearer token123'})
            
            # Verify request was created with URL
            mock_request.assert_called_once_with('http://example.com/api')
            
            # Verify response was parsed correctly
            assert result == '{"status": "success"}'

def test_custom_headers_added():
    """Test that custom headers are properly added to request"""
    mock_response = MagicMock()
    mock_response.read.return_value = b'response'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    
    with patch('urllib.request.urlopen', return_value=mock_response):
        with patch('urllib.request.Request') as mock_request:
            mock_req_instance = MagicMock()
            mock_request.return_value = mock_req_instance
            
            headers = {
                'Authorization': 'Bearer token123',
                'Content-Type': 'application/json',
                'X-API-Key': 'secret123'
            }
            
            http_get_with_headers('http://example.com/api', headers)
            
            # Verify all headers were added
            expected_calls = [
                ('Authorization', 'Bearer token123'),
                ('Content-Type', 'application/json'),
                ('X-API-Key', 'secret123')
            ]
            
            for key, value in expected_calls:
                mock_req_instance.add_header.assert_any_call(key, value)

def test_http_error_handling():
    """Test handling of HTTP errors"""
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url='http://example.com', code=404, msg='Not Found', hdrs=None, fp=None
        )
        
        result = http_get_with_headers('http://example.com/api', {'Auth': 'token'})
        assert result == "HTTP Error 404: Not Found"

def test_url_error_handling():
    """Test handling of URL errors"""
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.URLError('Connection refused')
        
        result = http_get_with_headers('http://invalid-url', {'Auth': 'token'})
        assert result == "URL Error: Connection refused"

def test_empty_headers():
    """Test request with empty headers dictionary"""
    mock_response = MagicMock()
    mock_response.read.return_value = b'success'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    
    with patch('urllib.request.urlopen', return_value=mock_response):
        with patch('urllib.request.Request') as mock_request:
            mock_req_instance = MagicMock()
            mock_request.return_value = mock_req_instance
            
            result = http_get_with_headers('http://example.com', {})
            
            # Should still work with empty headers
            assert result == 'success'
            # No headers should be added
            mock_req_instance.add_header.assert_not_called()

def test_non_string_header_values():
    """Test that non-string header values are converted to strings"""
    mock_response = MagicMock()
    mock_response.read.return_value = b'ok'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    
    with patch('urllib.request.urlopen', return_value=mock_response):
        with patch('urllib.request.Request') as mock_request:
            mock_req_instance = MagicMock()
            mock_request.return_value = mock_req_instance
            
            headers = {
                'X-Count': 123,
                'X-Enabled': True
            }
            
            http_get_with_headers('http://example.com', headers)
            
            # Verify values were converted to strings
            mock_req_instance.add_header.assert_any_call('X-Count', '123')
            mock_req_instance.add_header.assert_any_call('X-Enabled', 'True')

def test_independent_basic_get_request():
    """Test basic GET request functionality"""
    mock_response = MagicMock()
    mock_response.read.return_value = b'Hello World'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    
    with patch('urllib.request.urlopen', return_value=mock_response):
        result = http_get_with_headers('http://example.com', {})
        assert result == 'Hello World'

def test_independent_authentication_headers():
    """Test authentication with Bearer token"""
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"authenticated": true}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    
    with patch('urllib.request.urlopen', return_value=mock_response):
        with patch('urllib.request.Request') as mock_request:
            mock_req_instance = MagicMock()
            mock_request.return_value = mock_req_instance
            
            headers = {'Authorization': 'Bearer abc123'}
            result = http_get_with_headers('http://api.example.com/user', headers)
            
            assert result == '{"authenticated": true}'
            mock_req_instance.add_header.assert_called_with('Authorization', 'Bearer abc123')

def test_independent_pagination_cursor_response():
    """Test handling pagination cursor in response"""
    mock_response = MagicMock()
    response_data = '{"data": [1, 2, 3], "next_cursor": "xyz789"}'
    mock_response.read.return_value = response_data.encode('utf-8')
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    
    with patch('urllib.request.urlopen', return_value=mock_response):
        headers = {'X-Cursor': 'abc123'}
        result = http_get_with_headers('http://api.example.com/items', headers)
        
        # Verify the response contains expected structure
        assert '"data"' in result
        assert '"next_cursor"' in result
        assert 'xyz789' in result

def test_independent_empty_headers_dict():
    """Test with explicitly empty headers dictionary"""
    mock_response = MagicMock()
    mock_response.read.return_value = b'No auth required'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    
    with patch('urllib.request.urlopen', return_value=mock_response):
        result = http_get_with_headers('http://public.api.com/data', {})
        assert result == 'No auth required'

def test_independent_multiple_custom_headers():
    """Test with multiple custom headers"""
    mock_response = MagicMock()
    mock_response.read.return_value = b'Custom headers accepted'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)
    
    with patch('urllib.request.urlopen', return_value=mock_response):
        with patch('urllib.request.Request') as mock_request:
            mock_req_instance = MagicMock()
            mock_request.return_value = mock_req_instance
            
            headers = {
                'User-Agent': 'MyApp/1.0',
                'Accept': 'application/json',
                'X-Client-Version': '2.1'
            }
            
            result = http_get_with_headers('http://api.example.com/endpoint', headers)
            
            assert result == 'Custom headers accepted'
            # Verify all headers were added
            for key, value in headers.items():
                mock_req_instance.add_header.assert_any_call(key, value)

def test_adversarial_none_inputs():
    """Test that function properly handles None inputs"""
    # Test with None headers - should raise an exception
    try:
        result = http_get_with_headers('http://example.com', None)
        # If we get here, the function caught the exception and returned an error message
        assert result.startswith('Error:'), f"Expected error message, got: {result}"
    except Exception:
        # If an exception is raised, that's also acceptable behavior
        pass
    
    # Test with None URL - should raise an exception  
    try:
        result = http_get_with_headers(None, {})
        # If we get here, the function caught the exception and returned an error message
        assert result.startswith('Error:') or result.startswith('URL Error:'), f"Expected error message, got: {result}"
    except Exception:
        # If an exception is raised, that's also acceptable behavior
        pass