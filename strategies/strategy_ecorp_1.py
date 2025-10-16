# --- strategy_robust.py ---
# Assuming these imports are at the top of your script
import numpy as np
import pywt

def embedding(image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
    """
    Function to embed the watermark using a robust, 3-level DWT technique.
    It prioritizes robustness by embedding in low-frequency bands with redundancy.

    :param image: The original grayscale image as a NumPy array (uint8).
    :param watermark: The watermark as a 1D NumPy array of bits.
    :return: Watermarked image as a NumPy array (uint8).
    """
    WATERMARK_SIZE = 1024

    # 1. Convert image to float for processing
    image_float = image.astype(np.float32)
    
    # 2. Apply a 3-level DWT decomposition to access the most robust coefficients
    # Level 1
    coeffs1 = pywt.dwt2(image_float, 'db4', mode='symmetric')
    LL1, (LH1, HL1, HH1) = coeffs1
    
    # Level 2
    coeffs2 = pywt.dwt2(LL1, 'db4', mode='symmetric')
    LL2, (LH2, HL2, HH2) = coeffs2
    
    # Level 3
    coeffs3 = pywt.dwt2(LL2, 'db4', mode='symmetric')
    LL3, (LH3, HL3, HH3) = coeffs3
    
    # 3. Prepare embedding regions, prioritizing the most robust (lowest frequency)
    embed_regions = [LL3, LH2, HL2]
    
    # 4. Embed the watermark using quantization with redundancy
    watermark_idx = 0
    
    for region in embed_regions:
        if watermark_idx >= WATERMARK_SIZE:
            break
        
        region_flat = region.flatten()
        
        # Iterate in steps of 4, as each bit is embedded into 4 coefficients
        for i in range(0, len(region_flat) - 3, 4):
            if watermark_idx >= WATERMARK_SIZE:
                break

            bit = watermark[watermark_idx]
            
            # Embed the same bit into a block of 4 coefficients for redundancy
            for j in range(4):
                coeff = region_flat[i + j]
                
                # Use a larger quantization step for more robustness
                Q = max(2.0, abs(coeff) * 0.1)
                quantized = np.round(coeff / Q)
                
                if bit == 1: # Target: odd quantization
                    if int(quantized) % 2 == 0:
                        quantized += 1
                else: # Target: even quantization
                    if int(quantized) % 2 == 1:
                        # Move to the nearest even number (e.g., from 3 to 4, from -3 to -4)
                        quantized += 1 if quantized > 0 else -1

                region_flat[i + j] = quantized * Q

            watermark_idx += 1
        
        # Reshape the flattened array back to its original shape
        modified_region = region_flat.reshape(region.shape)
        region[:, :] = modified_region

    # 5. Reconstruct the image from the modified coefficients
    # Reconstruct level 2
    LL2_reconstructed = pywt.idwt2((LL3, (LH3, HL3, HH3)), 'db4', mode='symmetric')
    if LL2_reconstructed.shape != LL2.shape:
        LL2_reconstructed = LL2_reconstructed[:LL2.shape[0], :LL2.shape[1]]

    # Reconstruct level 1
    LL1_reconstructed = pywt.idwt2((LL2_reconstructed, (LH2, HL2, HH2)), 'db4', mode='symmetric')
    if LL1_reconstructed.shape != LL1.shape:
        LL1_reconstructed = LL1_reconstructed[:LL1.shape[0], :LL1.shape[1]]

    # Reconstruct final image
    watermarked_float = pywt.idwt2((LL1_reconstructed, (LH1, HL1, HH1)), 'db4', mode='symmetric')
    if watermarked_float.shape != image.shape:
        watermarked_float = watermarked_float[:image.shape[0], :image.shape[1]]

    # 6. Finalize and convert back to uint8
    watermarked_image = np.clip(watermarked_float, 0, 255).astype(np.uint8)
    
    return watermarked_image


def detection(original_image: np.ndarray, watermarked_image: np.ndarray) -> np.ndarray:
    """
    Detects and extracts a watermark embedded with the robust 3-level DWT technique.
    It uses a majority vote based on the 4-to-1 redundancy.

    :param original_image: The original grayscale image as a NumPy array (uint8).
    :param watermarked_image: The watermarked grayscale image as a NumPy array (uint8).
    :return: The extracted watermark as a 1D NumPy array of bits.
    """
    WATERMARK_SIZE = 1024
    extracted_watermark = []

    # 1. Convert both images to float for processing
    original_float = original_image.astype(np.float32)
    watermarked_float = watermarked_image.astype(np.float32)
    
    # 2. Apply the same 3-level DWT to both images
    # Original image (needed for calculating Q)
    coeffs1_orig = pywt.dwt2(original_float, 'db4', mode='symmetric')
    LL1_orig, _ = coeffs1_orig
    coeffs2_orig = pywt.dwt2(LL1_orig, 'db4', mode='symmetric')
    LL2_orig, (LH2_orig, HL2_orig, _) = coeffs2_orig
    coeffs3_orig = pywt.dwt2(LL2_orig, 'db4', mode='symmetric')
    LL3_orig, _ = coeffs3_orig

    # Watermarked image
    coeffs1_w = pywt.dwt2(watermarked_float, 'db4', mode='symmetric')
    LL1_w, _ = coeffs1_w
    coeffs2_w = pywt.dwt2(LL1_w, 'db4', mode='symmetric')
    LL2_w, (LH2_w, HL2_w, _) = coeffs2_w
    coeffs3_w = pywt.dwt2(LL2_w, 'db4', mode='symmetric')
    LL3_w, _ = coeffs3_w
    
    # 3. Prepare regions for extraction in the correct order
    original_regions = [LL3_orig, LH2_orig, HL2_orig]
    watermarked_regions = [LL3_w, LH2_w, HL2_w]
    
    # 4. Extract watermark by reversing the process
    watermark_idx = 0
    
    for orig_region, w_region in zip(original_regions, watermarked_regions):
        if watermark_idx >= WATERMARK_SIZE:
            break
        
        orig_flat = orig_region.flatten()
        w_flat = w_region.flatten()
        
        # Iterate in steps of 4 to read the redundant coefficients for each bit
        for i in range(0, len(w_flat) - 3, 4):
            if watermark_idx >= WATERMARK_SIZE:
                break
            
            votes = []
            # Read the 4 coefficients corresponding to a single watermark bit
            for j in range(4):
                original_coeff = orig_flat[i + j]
                watermarked_coeff = w_flat[i + j]

                # Recalculate Q using the original coefficient's value
                Q = max(2.0, abs(original_coeff) * 0.1)
                quantized = np.round(watermarked_coeff / Q)
                
                # Vote based on parity
                if int(quantized) % 2 == 0:
                    votes.append(0)
                else:
                    votes.append(1)
            
            # Majority vote decides the extracted bit
            if sum(votes) > 2:
                extracted_watermark.append(1)
            else:
                extracted_watermark.append(0)
            
            watermark_idx += 1
            
    return np.array(extracted_watermark, dtype=np.uint8)