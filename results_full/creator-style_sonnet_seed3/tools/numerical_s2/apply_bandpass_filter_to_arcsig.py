def apply_bandpass_filter_to_arcsig(arcsig_string, filter_spec):
    """
    Apply a band-pass filter to an ARCSIG signal using FFT.
    
    Utility: Filters an ARCSIG encoded signal to retain only frequencies within a specified band.
    Uses FFT to zero out frequency bins outside the passband, then reconstructs the signal.
    
    Args:
        arcsig_string (str): ARCSIG format string (e.g., "ARCSIG:v1;SR:100;LEN:128;ENC:f32le_b64;...")
        filter_spec (str): Filter specification in format "BANDPASS:<low_hz>,<high_hz>"
    
    Returns:
        str: Filtered ARCSIG string with same format but modified samples
    """
    import base64
    import struct
    import numpy as np
    
    # Parse filter specification
    if not filter_spec.startswith("BANDPASS:"):
        raise ValueError("Filter spec must start with 'BANDPASS:'")
    
    freq_range = filter_spec[9:]  # Remove "BANDPASS:" prefix
    low_hz, high_hz = map(float, freq_range.split(','))
    
    # Parse ARCSIG header and data
    parts = arcsig_string.split(';')
    if len(parts) < 5 or parts[0] != "ARCSIG:v1":
        raise ValueError("Invalid ARCSIG format")
    
    # Extract parameters
    sr = int(parts[1].split(':')[1])  # Sample rate
    length = int(parts[2].split(':')[1])  # Number of samples
    encoding = parts[3].split(':')[1]  # Should be f32le_b64
    data_b64 = parts[4]  # Base64 encoded data
    
    if encoding != "f32le_b64":
        raise ValueError("Only f32le_b64 encoding is supported")
    
    # Decode samples
    data_bytes = base64.b64decode(data_b64)
    samples = np.array([struct.unpack('<f', data_bytes[i:i+4])[0] for i in range(0, len(data_bytes), 4)])
    
    if len(samples) != length:
        raise ValueError(f"Expected {length} samples, got {len(samples)}")
    
    # Apply FFT
    fft_data = np.fft.fft(samples)
    
    # Create frequency bins
    freqs = np.fft.fftfreq(length, 1.0/sr)
    
    # Apply band-pass filter by zeroing out bins outside the passband
    filtered_fft = fft_data.copy()
    for i, freq in enumerate(freqs):
        if abs(freq) < low_hz or abs(freq) > high_hz:
            filtered_fft[i] = 0.0
    
    # Inverse FFT and take real part
    filtered_samples = np.real(np.fft.ifft(filtered_fft))
    
    # Re-encode samples to base64
    filtered_bytes = b''.join(struct.pack('<f', float(sample)) for sample in filtered_samples)
    filtered_b64 = base64.b64encode(filtered_bytes).decode('ascii')
    
    # Reconstruct ARCSIG string
    filtered_arcsig = f"ARCSIG:v1;SR:{sr};LEN:{length};ENC:f32le_b64;{filtered_b64}"
    
    return filtered_arcsig