def apply_bandpass_filter_to_arcsig(signal_string, filter_string):
    """
    Apply a band-pass filter to an ARCSIG signal using FFT-based frequency domain filtering.
    
    Utility: Filters an ARCSIG encoded signal to retain only frequencies within a specified band.
    Uses FFT to transform to frequency domain, zeros out bins outside the passband, then
    transforms back to time domain.
    
    Args:
        signal_string (str): ARCSIG formatted signal string (e.g., "ARCSIG:v1;SR:100;LEN:128;ENC:f32le_b64;...")
        filter_string (str): Filter specification in format "BANDPASS:<low_hz>,<high_hz>"
    
    Returns:
        str: Filtered ARCSIG string with same format as input but with filtered sample data
    """
    import numpy as np
    import base64
    import struct
    
    # Parse filter specification
    filter_parts = filter_string.split(':')
    if len(filter_parts) != 2 or filter_parts[0] != 'BANDPASS':
        raise ValueError("Filter must be in format 'BANDPASS:<low_hz>,<high_hz>'")
    
    freq_parts = filter_parts[1].split(',')
    if len(freq_parts) != 2:
        raise ValueError("Filter frequencies must be in format '<low_hz>,<high_hz>'")
    
    low_hz = float(freq_parts[0])
    high_hz = float(freq_parts[1])
    
    # Parse ARCSIG header
    parts = signal_string.split(';')
    if len(parts) < 5 or parts[0] != 'ARCSIG:v1':
        raise ValueError("Invalid ARCSIG format")
    
    sr = int(parts[1].split(':')[1])
    length = int(parts[2].split(':')[1])
    encoding = parts[3].split(':')[1]
    
    if encoding != 'f32le_b64':
        raise ValueError("Only f32le_b64 encoding is supported")
    
    # Decode base64 data
    b64_data = parts[4]
    raw_bytes = base64.b64decode(b64_data)
    
    # Convert to float32 samples
    samples = np.array(struct.unpack(f'<{length}f', raw_bytes), dtype=np.float32)
    
    # Apply FFT
    fft_samples = np.fft.fft(samples)
    
    # Create frequency bins
    freqs = np.fft.fftfreq(length, 1.0/sr)
    
    # Create bandpass mask - zero out frequencies outside [low_hz, high_hz]
    mask = (np.abs(freqs) >= low_hz) & (np.abs(freqs) <= high_hz)
    
    # Apply filter
    filtered_fft = fft_samples * mask
    
    # Inverse FFT and take real part
    filtered_samples = np.real(np.fft.ifft(filtered_fft)).astype(np.float32)
    
    # Re-encode to base64
    filtered_bytes = struct.pack(f'<{length}f', *filtered_samples)
    filtered_b64 = base64.b64encode(filtered_bytes).decode('ascii')
    
    # Reconstruct ARCSIG string
    filtered_arcsig = f"ARCSIG:v1;SR:{sr};LEN:{length};ENC:f32le_b64;{filtered_b64}"
    
    return filtered_arcsig