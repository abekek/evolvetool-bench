def decode_and_filter_arcsig_signal(arcsig_string, filter_spec):
    """
    Decode ARCSIG signal format, apply band-pass filter, and compute statistics.
    
    Utility: Parses ARCSIG encoded signal, applies specified band-pass filter to isolate 
    frequency components, then calculates mean, median, and standard deviation of filtered samples.
    
    Args:
        arcsig_string (str): ARCSIG formatted signal string with metadata and base64 encoded data
        filter_spec (str): Filter specification in format "BANDPASS:low_freq,high_freq"
    
    Returns:
        dict: Contains 'filtered_arcsig' (filtered signal in ARCSIG format) and 'stats' 
        (dict with mean, median, std of filtered time-domain samples)
    """
    import base64
    import struct
    import math
    import statistics
    
    # Parse ARCSIG header
    parts = arcsig_string.split(';')
    version = parts[0].split(':')[1]
    sample_rate = int(parts[1].split(':')[1])
    length = int(parts[2].split(':')[1])
    encoding = parts[3].split(':')[1]
    data_b64 = parts[4]
    
    # Decode base64 data
    data_bytes = base64.b64decode(data_b64)
    
    # Unpack float32 little-endian data
    samples = []
    for i in range(0, len(data_bytes), 4):
        value = struct.unpack('<f', data_bytes[i:i+4])[0]
        samples.append(value)
    
    # Parse filter specification
    filter_parts = filter_spec.split(':')
    filter_type = filter_parts[0]
    freq_range = filter_parts[1].split(',')
    low_freq = float(freq_range[0])
    high_freq = float(freq_range[1])
    
    # Apply simple band-pass filter using DFT/IDFT
    N = len(samples)
    
    # Forward DFT
    def dft(x):
        N = len(x)
        X = []
        for k in range(N):
            real = 0
            imag = 0
            for n in range(N):
                angle = -2 * math.pi * k * n / N
                real += x[n] * math.cos(angle)
                imag += x[n] * math.sin(angle)
            X.append(complex(real, imag))
        return X
    
    # Inverse DFT
    def idft(X):
        N = len(X)
        x = []
        for n in range(N):
            real = 0
            imag = 0
            for k in range(N):
                angle = 2 * math.pi * k * n / N
                real += X[k].real * math.cos(angle) - X[k].imag * math.sin(angle)
                imag += X[k].real * math.sin(angle) + X[k].imag * math.cos(angle)
            x.append(real / N)
        return x
    
    # Get frequency domain representation
    freq_domain = dft(samples)
    
    # Apply band-pass filter
    filtered_freq = []
    for k in range(len(freq_domain)):
        freq = k * sample_rate / N if k <= N//2 else (k - N) * sample_rate / N
        freq = abs(freq)
        
        if low_freq <= freq <= high_freq:
            filtered_freq.append(freq_domain[k])
        else:
            filtered_freq.append(complex(0, 0))
    
    # Convert back to time domain
    filtered_samples = idft(filtered_freq)
    
    # Convert filtered samples back to bytes
    filtered_bytes = b''
    for sample in filtered_samples:
        filtered_bytes += struct.pack('<f', float(sample))
    
    # Encode back to base64
    filtered_b64 = base64.b64encode(filtered_bytes).decode('ascii')
    
    # Reconstruct ARCSIG string
    filtered_arcsig = f"ARCSIG:{version};SR:{sample_rate};LEN:{length};ENC:{encoding};{filtered_b64}"
    
    # Calculate statistics
    mean_val = statistics.mean(filtered_samples)
    median_val = statistics.median(filtered_samples)
    std_val = statistics.stdev(filtered_samples) if len(filtered_samples) > 1 else 0.0
    
    stats = {
        'mean': mean_val,
        'median': median_val,
        'std': std_val
    }
    
    return {
        'filtered_arcsig': filtered_arcsig,
        'stats': stats
    }