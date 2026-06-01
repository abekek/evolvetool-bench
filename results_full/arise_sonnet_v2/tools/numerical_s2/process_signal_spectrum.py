def process_signal_spectrum(signal_string: str) -> list:
    """
    Decode ARCSIG signal format and compute its one-sided frequency spectrum.
    
    ARCSIG format: "ARCSIG:<base64_encoded_samples>:<sample_rate>"
    The base64 data contains float64 samples in little-endian format.
    
    Parameters:
    signal_string (str): ARCSIG formatted signal string
    
    Returns:
    list: One-sided frequency spectrum as [frequency, magnitude] pairs,
          or empty list if processing fails
    """
    import base64
    import struct
    import math
    
    try:
        # Handle empty or None input
        if not signal_string:
            return []
        
        # Parse ARCSIG format
        if not signal_string.startswith('ARCSIG:'):
            return []
        
        parts = signal_string.split(':') 
        if len(parts) != 3:
            return []
        
        _, encoded_data, sample_rate_str = parts
        
        try:
            sample_rate = float(sample_rate_str)
        except (ValueError, TypeError):
            return []
        
        if sample_rate <= 0:
            return []
        
        # Decode base64 data
        try:
            binary_data = base64.b64decode(encoded_data)
        except Exception:
            return []
        
        # Unpack float64 samples (little-endian)
        num_samples = len(binary_data) // 8
        if len(binary_data) % 8 != 0 or num_samples == 0:
            return []
        
        try:
            samples = list(struct.unpack(f'<{num_samples}d', binary_data))
        except struct.error:
            return []
        
        # Handle edge cases
        if not samples:
            return []
        
        # Validate samples - check for NaN and infinity
        for sample in samples:
            if math.isnan(sample) or math.isinf(sample):
                return []  # Return empty list if any sample is NaN or infinity
        
        # Implement FFT manually (Cooley-Tukey algorithm)
        def fft(x):
            N = len(x)
            if N <= 1:
                return x
            
            # Pad to next power of 2 for efficiency
            if N & (N - 1) != 0:
                next_pow2 = 1 << (N - 1).bit_length()
                x = x + [0] * (next_pow2 - N)
                N = next_pow2
            
            # Recursive FFT
            def _fft_recursive(x):
                N = len(x)
                if N <= 1:
                    return x
                
                # Divide
                even = _fft_recursive([x[i] for i in range(0, N, 2)])
                odd = _fft_recursive([x[i] for i in range(1, N, 2)])
                
                # Combine
                T = []
                for k in range(N // 2):
                    t = complex(math.cos(-2 * math.pi * k / N), math.sin(-2 * math.pi * k / N)) * odd[k]
                    T.append(t)
                
                return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]
            
            # Convert to complex
            x_complex = [complex(val, 0) for val in x]
            return _fft_recursive(x_complex)
        
        # Compute FFT
        fft_result = fft(samples)
        N = len(fft_result)
        
        # Compute one-sided spectrum (positive frequencies only)
        # Take first half + DC component
        one_sided_length = N // 2 + 1
        
        spectrum = []
        for i in range(one_sided_length):
            # Frequency bin
            freq = i * sample_rate / N
            
            # Magnitude (absolute value of complex number)
            magnitude = math.sqrt(fft_result[i].real**2 + fft_result[i].imag**2)
            
            # Scale appropriately for one-sided spectrum
            if i == 0 or (N % 2 == 0 and i == N // 2):
                # DC and Nyquist components (if present) - no scaling
                scaled_magnitude = magnitude / N
            else:
                # Other frequencies - multiply by 2 for one-sided
                scaled_magnitude = 2 * magnitude / N
            
            spectrum.append([freq, scaled_magnitude])
        
        return spectrum
        
    except Exception:
        return []