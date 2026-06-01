def decode_arcsig_and_compute_spectrum(arcsig_signal):
    """
    Decode an ARCSIG signal and compute its frequency spectrum with DC offset handling.
    
    Utility: Parses ARCSIG format signals, decodes base64-encoded float32 data, 
    and computes FFT spectrum while safely handling large DC offsets to prevent 
    overflow or NaN values.
    
    Args:
        arcsig_signal (str): ARCSIG format string containing signal parameters and data
        
    Returns:
        list: List of dictionaries with keys 'freq_hz' and 'magnitude' representing
              the one-sided frequency spectrum from DC to Nyquist frequency
    """
    import base64
    import struct
    import math
    
    # Parse ARCSIG header
    parts = arcsig_signal.split(';')
    if not parts[0].startswith('ARCSIG:'):
        raise ValueError("Invalid ARCSIG format")
    
    # Extract parameters
    sample_rate = None
    length = None
    data_b64 = None
    
    for part in parts:
        if part.startswith('SR:'):
            sample_rate = int(part.split(':')[1])
        elif part.startswith('LEN:'):
            length = int(part.split(':')[1])
        elif part.startswith('ENC:'):
            encoding = part.split(':')[1]
            if encoding != 'f32le_b64':
                raise ValueError("Unsupported encoding")
        elif not part.startswith(('ARCSIG:', 'SR:', 'LEN:', 'ENC:')):
            data_b64 = part
    
    if sample_rate is None or length is None or data_b64 is None:
        raise ValueError("Missing required ARCSIG parameters")
    
    # Decode base64 data
    binary_data = base64.b64decode(data_b64)
    
    # Unpack float32 little-endian data
    signal_data = []
    for i in range(0, len(binary_data), 4):
        float_val = struct.unpack('<f', binary_data[i:i+4])[0]
        signal_data.append(float_val)
    
    if len(signal_data) != length:
        raise ValueError(f"Data length mismatch: expected {length}, got {len(signal_data)}")
    
    # Remove DC offset to prevent overflow in FFT
    dc_offset = sum(signal_data) / len(signal_data)
    signal_no_dc = [x - dc_offset for x in signal_data]
    
    # Compute FFT manually (simple DFT implementation)
    N = len(signal_no_dc)
    fft_result = []
    
    for k in range(N // 2 + 1):  # One-sided spectrum
        real_sum = 0.0
        imag_sum = 0.0
        
        for n in range(N):
            angle = -2.0 * math.pi * k * n / N
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)
            
            real_sum += signal_no_dc[n] * cos_val
            imag_sum += signal_no_dc[n] * sin_val
        
        # Calculate magnitude
        magnitude = math.sqrt(real_sum * real_sum + imag_sum * imag_sum) / N
        
        # Scale non-DC bins for one-sided spectrum
        if k > 0 and k < N // 2:
            magnitude *= 2
        
        # Add back DC offset magnitude for DC bin
        if k == 0:
            magnitude = abs(dc_offset)
        
        # Calculate frequency
        freq_hz = k * sample_rate / N
        
        fft_result.append({
            "freq_hz": round(freq_hz, 6),
            "magnitude": round(magnitude, 6)
        })
    
    return fft_result