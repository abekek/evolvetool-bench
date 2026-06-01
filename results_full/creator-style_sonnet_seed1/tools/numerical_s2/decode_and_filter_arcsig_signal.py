def decode_and_filter_arcsig_signal(arcsig_string, filter_spec):
    """
    Decode ARCSIG signal, apply band-pass filter, and compute statistics.
    
    Utility: Parses ARCSIG encoded signal data, applies a simple band-pass filter
    to isolate frequency components, and computes mean, median, and standard 
    deviation of the filtered time-domain samples.
    
    Args:
        arcsig_string (str): ARCSIG formatted signal string with metadata and base64 encoded data
        filter_spec (str): Filter specification in format "BANDPASS:low_freq,high_freq"
    
    Returns:
        dict: JSON object with 'filtered_arcsig' (filtered signal as ARCSIG string) 
              and 'stats' (dict with mean, median, std of filtered samples)
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
    b64_data = parts[4]
    
    # Decode base64 data
    raw_data = base64.b64decode(b64_data)
    
    # Unpack float32 little-endian data
    samples = []
    for i in range(0, len(raw_data), 4):
        value = struct.unpack('<f', raw_data[i:i+4])[0]
        samples.append(value)
    
    # Parse filter specification
    filter_parts = filter_spec.split(':')
    freq_range = filter_parts[1].split(',')
    low_freq = float(freq_range[0])
    high_freq = float(freq_range[1])
    
    # Apply simple frequency domain filter using DFT
    N = len(samples)
    
    # Forward DFT
    freq_domain = []
    for k in range(N):
        real_sum = 0
        imag_sum = 0
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            real_sum += samples[n] * math.cos(angle)
            imag_sum += samples[n] * math.sin(angle)
        freq_domain.append((real_sum, imag_sum))
    
    # Apply band-pass filter in frequency domain
    filtered_freq = []
    for k in range(N):
        freq_hz = k * sample_rate / N
        if k > N // 2:
            freq_hz = (k - N) * sample_rate / N
            freq_hz = abs(freq_hz)
        
        if low_freq <= freq_hz <= high_freq:
            filtered_freq.append(freq_domain[k])
        else:
            filtered_freq.append((0, 0))
    
    # Inverse DFT
    filtered_samples = []
    for n in range(N):
        real_sum = 0
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            real_part = filtered_freq[k][0] * math.cos(angle) - filtered_freq[k][1] * math.sin(angle)
            real_sum += real_part
        filtered_samples.append(real_sum / N)
    
    # Compute statistics
    mean_val = statistics.mean(filtered_samples)
    median_val = statistics.median(filtered_samples)
    std_val = statistics.stdev(filtered_samples) if len(filtered_samples) > 1 else 0
    
    # Encode filtered samples back to ARCSIG format
    filtered_raw = b''
    for sample in filtered_samples:
        filtered_raw += struct.pack('<f', sample)
    
    filtered_b64 = base64.b64encode(filtered_raw).decode('ascii')
    filtered_arcsig = f"ARCSIG:{version};SR:{sample_rate};LEN:{length};ENC:{encoding};{filtered_b64}"
    
    return {
        'filtered_arcsig': filtered_arcsig,
        'stats': {
            'mean': mean_val,
            'median': median_val,
            'std': std_val
        }
    }