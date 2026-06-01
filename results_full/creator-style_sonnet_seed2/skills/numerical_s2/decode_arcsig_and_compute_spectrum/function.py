def decode_arcsig_and_compute_spectrum(arcsig_string):
    """
    Decode an ARCSIG signal and compute its frequency spectrum using FFT.
    
    Utility: Parses ARCSIG format, decodes base64 encoded float32 data, removes DC offset 
    to prevent overflow, computes FFT magnitude spectrum, and returns frequency bins with magnitudes.
    
    Args:
        arcsig_string (str): ARCSIG formatted string containing signal metadata and base64 encoded data
        
    Returns:
        list: List of dictionaries with keys 'freq_hz' and 'magnitude' representing the frequency spectrum
    """
    import base64
    import struct
    import math
    
    # Parse ARCSIG format
    parts = arcsig_string.split(';')
    metadata = {}
    signal_data = None
    
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            if key == 'SR':
                metadata['sample_rate'] = int(value)
            elif key == 'LEN':
                metadata['length'] = int(value)
            elif key == 'ENC':
                metadata['encoding'] = value
        else:
            # This should be the base64 encoded data
            signal_data = part
    
    # Decode base64 data
    raw_bytes = base64.b64decode(signal_data)
    
    # Unpack float32 little endian data
    sample_count = len(raw_bytes) // 4
    samples = []
    for i in range(sample_count):
        float_bytes = raw_bytes[i*4:(i+1)*4]
        float_val = struct.unpack('<f', float_bytes)[0]
        samples.append(float_val)
    
    # Calculate DC component first (before removing it)
    dc_magnitude = abs(sum(samples) / len(samples))
    
    # Remove DC offset to prevent overflow in FFT
    mean_value = sum(samples) / len(samples)
    samples_no_dc = [s - mean_value for s in samples]
    
    # Compute FFT manually (DFT implementation)
    N = len(samples_no_dc)
    fft_result = []
    
    for k in range(N):
        real_sum = 0.0
        imag_sum = 0.0
        
        for n in range(N):
            angle = -2.0 * math.pi * k * n / N
            real_sum += samples_no_dc[n] * math.cos(angle)
            imag_sum += samples_no_dc[n] * math.sin(angle)
        
        magnitude = math.sqrt(real_sum * real_sum + imag_sum * imag_sum)
        fft_result.append(magnitude)
    
    # Create frequency bins
    sample_rate = metadata['sample_rate']
    spectrum = []
    
    # Only return first half of spectrum (positive frequencies) + DC
    for i in range(N // 2 + 1):
        freq_hz = i * sample_rate / N
        
        if i == 0:
            # DC component - use original DC magnitude
            magnitude = dc_magnitude
        else:
            # Other frequencies - normalize by length
            magnitude = fft_result[i] / N
        
        spectrum.append({
            "freq_hz": freq_hz,
            "magnitude": magnitude
        })
    
    return spectrum