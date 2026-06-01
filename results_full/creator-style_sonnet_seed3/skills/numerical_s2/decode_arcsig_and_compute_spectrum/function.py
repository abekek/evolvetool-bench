def decode_arcsig_and_compute_spectrum(arcsig_signal):
    """
    Decode an ARCSIG signal and compute its one-sided frequency spectrum.
    
    Utility:
        Parses ARCSIG format signals, decodes the base64-encoded data,
        and computes the magnitude spectrum using FFT. Returns frequency
        bins and their corresponding magnitudes as a JSON-formatted list.
    
    Args:
        arcsig_signal (str): ARCSIG formatted signal string containing
                           version, sample rate, length, encoding, and data
    
    Returns:
        str: JSON string containing list of {"freq_hz": float, "magnitude": float}
             objects representing the one-sided frequency spectrum
    """
    import base64
    import struct
    import math
    import json
    
    # Parse ARCSIG header
    parts = arcsig_signal.split(';')
    header_info = {}
    data_b64 = None
    
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            if key in ['SR', 'LEN']:
                header_info[key] = int(value)
            else:
                header_info[key] = value
        else:
            # This is the base64 data
            data_b64 = part
    
    # Decode base64 data
    binary_data = base64.b64decode(data_b64)
    
    # Unpack float32 little-endian data
    sample_count = len(binary_data) // 4
    samples = struct.unpack(f'<{sample_count}f', binary_data)
    
    # Get parameters
    sample_rate = header_info['SR']
    N = len(samples)
    
    # Compute FFT manually (DFT implementation)
    def compute_dft(x):
        N = len(x)
        X = []
        for k in range(N):
            real_sum = 0.0
            imag_sum = 0.0
            for n in range(N):
                angle = -2 * math.pi * k * n / N
                real_sum += x[n] * math.cos(angle)
                imag_sum += x[n] * math.sin(angle)
            X.append(complex(real_sum, imag_sum))
        return X
    
    # Compute DFT
    fft_result = compute_dft(samples)
    
    # Compute one-sided spectrum (only positive frequencies)
    one_sided_length = N // 2 + 1
    spectrum = []
    
    for i in range(one_sided_length):
        # Frequency bin
        freq = i * sample_rate / N
        
        # Magnitude
        magnitude = abs(fft_result[i]) / N
        
        # Scale non-DC components for one-sided spectrum
        if i > 0 and i < N // 2:
            magnitude *= 2
        
        spectrum.append({
            "freq_hz": round(freq, 2),
            "magnitude": round(magnitude, 4)
        })
    
    return json.dumps(spectrum)