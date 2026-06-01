def decode_arcsig_and_compute_spectrum(arcsig_string):
    """
    Decodes an ARCSIG signal and computes its frequency spectrum.
    
    Utility: Parses ARCSIG format signals, decodes base64-encoded float32 data,
    and computes the one-sided frequency spectrum using FFT.
    
    Args:
        arcsig_string (str): ARCSIG formatted signal string containing metadata and encoded data
        
    Returns:
        list: List of dictionaries with keys "freq_hz" and "magnitude" representing the frequency spectrum
    """
    import base64
    import struct
    import math
    import json
    
    # Parse ARCSIG header
    parts = arcsig_string.split(';')
    header = {}
    
    for part in parts[:-1]:  # All parts except the last one (which is data)
        if ':' in part:
            key, value = part.split(':', 1)
            if key in ['SR', 'LEN']:
                header[key] = int(value)
            else:
                header[key] = value
    
    # Extract base64 data (last part)
    b64_data = parts[-1]
    
    # Decode base64 to bytes
    binary_data = base64.b64decode(b64_data)
    
    # Decode float32 little-endian data
    num_samples = len(binary_data) // 4
    signal = []
    for i in range(num_samples):
        float_bytes = binary_data[i*4:(i+1)*4]
        float_val = struct.unpack('<f', float_bytes)[0]
        signal.append(float_val)
    
    # Get parameters
    sample_rate = header['SR']
    N = len(signal)
    
    # Compute FFT manually (since we can't import numpy)
    def compute_fft(x):
        N = len(x)
        if N <= 1:
            return x
        
        # Simple DFT for small sizes or non-power-of-2
        result = []
        for k in range(N):
            sum_real = 0
            sum_imag = 0
            for n in range(N):
                angle = -2 * math.pi * k * n / N
                sum_real += x[n] * math.cos(angle)
                sum_imag += x[n] * math.sin(angle)
            result.append(complex(sum_real, sum_imag))
        return result
    
    # Compute FFT
    fft_result = compute_fft(signal)
    
    # Compute one-sided spectrum (only positive frequencies)
    spectrum = []
    for k in range(N // 2 + 1):
        freq_hz = k * sample_rate / N
        magnitude = abs(fft_result[k]) / N
        
        # Double the magnitude for positive frequencies (except DC and Nyquist)
        if k > 0 and k < N // 2:
            magnitude *= 2
            
        spectrum.append({
            "freq_hz": round(freq_hz, 2),
            "magnitude": round(magnitude, 4)
        })
    
    return spectrum