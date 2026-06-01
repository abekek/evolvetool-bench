def apply_bandpass_filter_to_arcsig(arcsig_string, filter_spec):
    """
    Apply a band-pass filter to an ARCSIG signal using FFT-based filtering.
    
    Utility: Filters an ARCSIG encoded signal to retain only frequencies within 
    the specified band-pass range. Uses FFT to zero out frequency bins outside 
    the pass band, then reconstructs the time-domain signal.
    
    Args:
        arcsig_string (str): Input ARCSIG formatted signal string
        filter_spec (str): Filter specification in format "BANDPASS:<low_hz>,<high_hz>"
    
    Returns:
        str: Filtered ARCSIG string with same format but filtered samples
    """
    import base64
    import struct
    import cmath
    import math
    
    # Parse filter specification
    filter_parts = filter_spec.split(':')
    if len(filter_parts) != 2 or filter_parts[0] != 'BANDPASS':
        raise ValueError("Invalid filter spec format")
    
    freq_range = filter_parts[1].split(',')
    if len(freq_range) != 2:
        raise ValueError("Invalid frequency range format")
    
    low_hz = float(freq_range[0])
    high_hz = float(freq_range[1])
    
    # Parse ARCSIG header
    parts = arcsig_string.split(';')
    if len(parts) != 5 or parts[0] != 'ARCSIG:v1':
        raise ValueError("Invalid ARCSIG format")
    
    sr = int(parts[1].split(':')[1])
    length = int(parts[2].split(':')[1])
    encoding = parts[3].split(':')[1]
    data_b64 = parts[4]
    
    if encoding != 'f32le_b64':
        raise ValueError("Unsupported encoding")
    
    # Decode samples
    data_bytes = base64.b64decode(data_b64)
    samples = []
    for i in range(0, len(data_bytes), 4):
        sample_bytes = data_bytes[i:i+4]
        sample = struct.unpack('<f', sample_bytes)[0]
        samples.append(sample)
    
    # Compute FFT manually
    n = len(samples)
    fft_result = []
    
    for k in range(n):
        real_sum = 0.0
        imag_sum = 0.0
        for j in range(n):
            angle = -2.0 * math.pi * k * j / n
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)
            real_sum += samples[j] * cos_val
            imag_sum += samples[j] * sin_val
        fft_result.append(complex(real_sum, imag_sum))
    
    # Apply band-pass filter
    filtered_fft = []
    for k in range(n):
        # Calculate frequency for this bin
        if k <= n // 2:
            freq = k * sr / n
        else:
            freq = (k - n) * sr / n
        
        # Zero out bins outside the pass band
        if abs(freq) < low_hz or abs(freq) > high_hz:
            filtered_fft.append(complex(0.0, 0.0))
        else:
            filtered_fft.append(fft_result[k])
    
    # Compute IFFT manually
    filtered_samples = []
    for j in range(n):
        real_sum = 0.0
        imag_sum = 0.0
        for k in range(n):
            angle = 2.0 * math.pi * k * j / n
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)
            fft_real = filtered_fft[k].real
            fft_imag = filtered_fft[k].imag
            real_sum += fft_real * cos_val - fft_imag * sin_val
            imag_sum += fft_real * sin_val + fft_imag * cos_val
        
        # Take real part and normalize
        filtered_samples.append(real_sum / n)
    
    # Re-encode as ARCSIG
    filtered_bytes = b''
    for sample in filtered_samples:
        filtered_bytes += struct.pack('<f', sample)
    
    filtered_b64 = base64.b64encode(filtered_bytes).decode('ascii')
    
    # Reconstruct ARCSIG string
    filtered_arcsig = f"ARCSIG:v1;SR:{sr};LEN:{length};ENC:f32le_b64;{filtered_b64}"
    
    return filtered_arcsig