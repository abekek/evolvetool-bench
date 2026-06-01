def apply_bandpass_filter_to_arcsig(arcsig_signal, filter_spec):
    """
    Apply a band-pass filter to an ARCSIG signal using FFT-based filtering.
    
    Utility:
        Decodes an ARCSIG signal, applies a band-pass filter in the frequency domain
        by zeroing out FFT bins outside the specified frequency range, and re-encodes
        the filtered signal back to ARCSIG format.
    
    Args:
        arcsig_signal (str): ARCSIG formatted signal string
        filter_spec (str): Filter specification in format "BANDPASS:<low_hz>,<high_hz>"
    
    Returns:
        str: Filtered ARCSIG signal string with same format as input
    """
    import base64
    import struct
    import numpy as np
    
    # Parse filter specification
    if not filter_spec.startswith("BANDPASS:"):
        raise ValueError("Invalid filter specification format")
    
    freq_range = filter_spec.split(":")[1]
    low_hz, high_hz = map(float, freq_range.split(","))
    
    # Parse ARCSIG header
    parts = arcsig_signal.split(";")
    header_info = {}
    
    for part in parts[:-1]:  # Skip the data part
        if ":" in part:
            key, value = part.split(":", 1)
            header_info[key] = value
    
    # Extract metadata
    sample_rate = int(header_info["SR"])
    length = int(header_info["LEN"])
    encoding = header_info["ENC"]
    
    if encoding != "f32le_b64":
        raise ValueError("Only f32le_b64 encoding is supported")
    
    # Decode base64 data
    data_b64 = parts[-1]
    data_bytes = base64.b64decode(data_b64)
    
    # Unpack float32 little-endian samples
    samples = np.array(struct.unpack(f"<{length}f", data_bytes))
    
    # Compute FFT
    fft_samples = np.fft.fft(samples)
    
    # Create frequency bins
    freqs = np.fft.fftfreq(length, 1.0 / sample_rate)
    
    # Apply band-pass filter by zeroing out bins outside [low_hz, high_hz]
    # Handle both positive and negative frequencies
    mask = (np.abs(freqs) >= low_hz) & (np.abs(freqs) <= high_hz)
    filtered_fft = fft_samples * mask
    
    # Compute IFFT and take real part
    filtered_samples = np.real(np.fft.ifft(filtered_fft))
    
    # Re-encode to bytes
    filtered_bytes = struct.pack(f"<{length}f", *filtered_samples.astype(np.float32))
    
    # Encode to base64
    filtered_b64 = base64.b64encode(filtered_bytes).decode('ascii')
    
    # Reconstruct ARCSIG string
    filtered_arcsig = f"ARCSIG:v1;SR:{sample_rate};LEN:{length};ENC:{encoding};{filtered_b64}"
    
    return filtered_arcsig