def embedding(input1: str, input2: str):
    """
    Function to embed the watermark in the image using robust DWT-based technique

    :param input1: Name of the original image file
    :param input2: Name of the watermark file (not used - we generate pseudo-random watermark)
    :return: Watermarked image
    """
    import cv2
    import numpy as np
    import pywt
    
    FIXED_SEED = 42
    WATERMARK_SIZE = 1024
    
    # Load the image
    image = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {input1}")
    
    # Generate pseudo-random watermark
    np.random.seed(FIXED_SEED)
    watermark = np.random.randint(0, 2, size=WATERMARK_SIZE)
    
    # Convert to float for processing
    image_float = image.astype(np.float32)
    
    # Apply single-level DWT decomposition
    coeffs = pywt.dwt2(image_float, 'db4', mode='symmetric')
    LL, (LH, HL, HH) = coeffs
    
    # Apply second level DWT on LL
    coeffs2 = pywt.dwt2(LL, 'db4', mode='symmetric')
    LL2, (LH2, HL2, HH2) = coeffs2
    
    # Apply third level DWT on LL2 for maximum robustness
    coeffs3 = pywt.dwt2(LL2, 'db4', mode='symmetric')
    LL3, (LH3, HL3, HH3) = coeffs3
    
    # Prepare embedding regions (prioritize low-frequency components)
    embed_regions = [
        LL3.flatten(),    # Highest priority - most robust
        LH2.flatten(),    # Medium priority
        HL2.flatten()     # Medium priority
    ]
    
    # Store original shapes for reconstruction
    original_shapes = [LL3.shape, LH2.shape, HL2.shape]
    
    # Calculate embedding strength based on local variance
    def calculate_embedding_strength(coeff_value, base_strength=0.05):
        """Calculate adaptive embedding strength"""
        if abs(coeff_value) > 50:
            return base_strength * 1.5
        elif abs(coeff_value) > 20:
            return base_strength * 1.2
        else:
            return base_strength * 0.8
    
    # Embed watermark using quantization-based method
    watermark_idx = 0
    
    for region_idx, region in enumerate(embed_regions):
        if watermark_idx >= WATERMARK_SIZE:
            break
        
        region_size = len(region)
        # Calculate how many bits we can embed in this region
        bits_per_region = min(WATERMARK_SIZE - watermark_idx, region_size // 4)
        
        for i in range(0, bits_per_region * 4, 4):
            if watermark_idx >= WATERMARK_SIZE or i >= region_size:
                break
            
            # Get watermark bit
            bit = watermark[watermark_idx]
            
            # Embed bit with redundancy across 4 coefficients
            for j in range(4):
                if i + j < region_size:
                    coeff = region[i + j]
                    alpha = calculate_embedding_strength(coeff)
                    
                    # Quantization-based embedding
                    Q = max(2.0, abs(coeff) * 0.1)
                    
                    if bit == 1:
                        # Make coefficient correspond to odd quantization
                        quantized = np.round(coeff / Q)
                        if int(quantized) % 2 == 0:
                            region[i + j] = (quantized + 1) * Q * (1 + alpha)
                        else:
                            region[i + j] = quantized * Q * (1 + alpha)
                    else:
                        # Make coefficient correspond to even quantization
                        quantized = np.round(coeff / Q)
                        if int(quantized) % 2 == 1:
                            region[i + j] = (quantized + 1) * Q * (1 + alpha)
                        else:
                            region[i + j] = quantized * Q * (1 + alpha)
            
            watermark_idx += 1
    
    # Reconstruct the modified coefficient arrays
    LL3_modified = embed_regions[0].reshape(original_shapes[0])
    LH2_modified = embed_regions[1].reshape(original_shapes[1])
    HL2_modified = embed_regions[2].reshape(original_shapes[2])
    
    # Reconstruct level 3
    LL2_reconstructed = pywt.idwt2((LL3_modified, (LH3, HL3, HH3)), 'db4', mode='symmetric')
    
    # Ensure LL2_reconstructed has the right size for level 2 reconstruction
    if LL2_reconstructed.shape != LL2.shape:
        # Crop or pad to match original size
        min_h, min_w = min(LL2_reconstructed.shape[0], LL2.shape[0]), min(LL2_reconstructed.shape[1], LL2.shape[1])
        LL2_temp = np.zeros(LL2.shape)
        LL2_temp[:min_h, :min_w] = LL2_reconstructed[:min_h, :min_w]
        LL2_reconstructed = LL2_temp
    
    # Reconstruct level 2
    LL_reconstructed = pywt.idwt2((LL2_reconstructed, (LH2_modified, HL2_modified, HH2)), 'db4', mode='symmetric')
    
    # Ensure LL_reconstructed has the right size for level 1 reconstruction
    if LL_reconstructed.shape != LL.shape:
        min_h, min_w = min(LL_reconstructed.shape[0], LL.shape[0]), min(LL_reconstructed.shape[1], LL.shape[1])
        LL_temp = np.zeros(LL.shape)
        LL_temp[:min_h, :min_w] = LL_reconstructed[:min_h, :min_w]
        LL_reconstructed = LL_temp
    
    # Reconstruct level 1 (final image)
    watermarked_float = pywt.idwt2((LL_reconstructed, (LH, HL, HH)), 'db4', mode='symmetric')
    
    # Handle size mismatch with original image
    if watermarked_float.shape != image_float.shape:
        min_h, min_w = min(watermarked_float.shape[0], image_float.shape[0]), min(watermarked_float.shape[1], image_float.shape[1])
        watermarked_temp = np.zeros(image_float.shape)
        watermarked_temp[:min_h, :min_w] = watermarked_float[:min_h, :min_w]
        watermarked_float = watermarked_temp
    
    # Ensure values are in valid range and convert back to uint8
    watermarked_float = np.clip(watermarked_float, 0, 255)
    watermarked_image = watermarked_float.astype(np.uint8)
    
    # Add subtle noise for anti-detection
    np.random.seed(FIXED_SEED + 1)
    noise = np.random.normal(0, 0.3, watermarked_image.shape)
    watermarked_image = np.clip(watermarked_image + noise, 0, 255).astype(np.uint8)
    
    return watermarked_image

