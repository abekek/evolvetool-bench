def decode_arcsig_and_compute_spectrum(arcsig_signal: str) -> list[dict[str, float]]:
    """Decode ARCSIG signal and compute frequency spectrum with robust DC offset handling."""
    import base64
    import struct
    import math
    import traceback
    import sys
    
    try:
        # Step 1: Parse ARCSIG header to extract sample rate, length, and encoding parameters
        parts = arcsig_signal.split(';')
        header = {}
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                header[key] = value
        
        sample_rate = int(header['SR'])
        length = int(header['LEN'])
        encoding = header['ENC']
        
        # Find the base64 data (after the last semicolon)
        data_start = arcsig_signal.rfind(';') + 1
        b64_data = arcsig_signal[data_start:]
        
        # Step 2: Decode base64 data and convert to float32 array using little-endian byte order
        decoded_bytes = base64.b64decode(b64_data)
        
        # Unpack as little-endian float32 values
        float_data = []
        for i in range(0, len(decoded_bytes), 4):
            if i + 4 <= len(decoded_bytes):
                value = struct.unpack('<f', decoded_bytes[i:i+4])[0]
                float_data.append(value)
        
        # Ensure we have the expected length
        signal = float_data[:length]
        
        # Step 3: Remove DC offset by subtracting mean value to prevent overflow and improve numerical stability
        mean_value = sum(signal) / len(signal)
        dc_offset_removed = [x - mean_value for x in signal]
        
        # Step 4: Apply FFT to compute frequency spectrum, using appropriate windowing if needed
        # Implement DFT manually since we're using only standard library
        N = len(dc_offset_removed)
        spectrum = []
        
        for k in range(N // 2 + 1):  # Only compute positive frequencies + DC
            real_sum = 0.0
            imag_sum = 0.0
            
            for n in range(N):
                angle = -2.0 * math.pi * k * n / N
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)
                
                real_sum += dc_offset_removed[n] * cos_val
                imag_sum += dc_offset_removed[n] * sin_val
            
            spectrum.append((real_sum, imag_sum))
        
        # Step 5: Calculate frequency bins based on sample rate and FFT length
        freq_bins = []
        for k in range(len(spectrum)):
            freq_hz = k * sample_rate / N
            freq_bins.append(freq_hz)
        
        # Step 6: Compute magnitudes with numerical safeguards and format as JSON-compatible list of frequency-magnitude pairs
        result = []
        for i, (real, imag) in enumerate(spectrum):
            freq_hz = freq_bins[i]
            
            # For DC bin, add back the mean to get the original DC magnitude
            if i == 0:
                magnitude = abs(mean_value)
            else:
                # Compute magnitude with numerical safeguards
                magnitude_squared = real * real + imag * imag
                magnitude = math.sqrt(max(0.0, magnitude_squared)) / N
            
            # Ensure no NaN or overflow
            if math.isnan(magnitude) or math.isinf(magnitude):
                magnitude = 0.0
            
            result.append({"freq_hz": freq_hz, "magnitude": magnitude})
        
        return result
        
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        return []