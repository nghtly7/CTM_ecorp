# Assuming these imports are at the top of your script
import numpy as np
import pywt


def embedding(image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
    """
    Function to embed the watermark in the image using a 2-level DWT.
    Optimized for high WPSNR while maintaining robustness.

    :param image: The original grayscale image as a NumPy array (uint8).
    :param watermark: The watermark as a 1D NumPy array of bits.
    :return: Watermarked image as a NumPy array (uint8).
    """
    WATERMARK_SIZE = 1024  # This should match the length of your watermark array

    # 1. Convert image to float for processing
    image_float = image.astype(np.float64)

    # 2. Apply a 2-level DWT decomposition
    # Level 1
    coeffs1 = pywt.dwt2(image_float, 'db4', mode='symmetric')
    LL1, (LH1, HL1, HH1) = coeffs1

    # Level 2
    coeffs2 = pywt.dwt2(LL1, 'db4', mode='symmetric')
    LL2, (LH2, HL2, HH2) = coeffs2

    # 3. Prepare for embedding in mid-frequency coefficients (LH2, HL2)
    embed_coeffs = [LH2.copy(), HL2.copy()]

    # 4. Embed the watermark using a quantization-based method
    watermark_idx = 0
    block_size = 8

    # --- FIX 1: Iterate using an index to modify the list elements in place ---
    for idx in range(len(embed_coeffs)):
        if watermark_idx >= WATERMARK_SIZE:
            break

        # Iterate over the coefficient array in small blocks
        for i in range(0, embed_coeffs[idx].shape[0], block_size):
            for j in range(0, embed_coeffs[idx].shape[1], block_size):
                if watermark_idx >= WATERMARK_SIZE:
                    break

                # Extract a block from the array in the list
                end_i = min(i + block_size, embed_coeffs[idx].shape[0])
                end_j = min(j + block_size, embed_coeffs[idx].shape[1])
                block = embed_coeffs[idx][i:end_i, j:end_j]

                # Embed watermark bits into the flattened block
                block_flat = block.flatten()

                for k in range(min(len(block_flat), WATERMARK_SIZE - watermark_idx)):
                    bit = watermark[watermark_idx + k]
                    coeff_val = block_flat[k]

                    # Quantization Index Modulation (QIM)
                    if abs(coeff_val) > 1e-6:
                        Q = max(0.1, abs(coeff_val) * 0.01)
                        quantized = np.round(coeff_val / Q)

                        # Embed bit by forcing the quantized value to be even or odd
                        if bit == 1:  # Target: odd
                            if int(quantized) % 2 == 0:
                                quantized += 1 if coeff_val >= 0 else -1
                        else:  # Target: even
                            if int(quantized) % 2 == 1:
                                quantized += 1 if coeff_val >= 0 else -1

                        # Apply the definitive modification
                        new_val = quantized * Q
                        block_flat[k] = new_val

                watermark_idx += min(len(block_flat), WATERMARK_SIZE - watermark_idx)

                # --- FIX 2: Place the modified block back into the ORIGINAL array in the list ---
                block_reshaped = block_flat.reshape(block.shape)
                embed_coeffs[idx][i:end_i, j:end_j] = block_reshaped

    # 5. Reconstruct the image by performing inverse DWT
    LH2_modified, HL2_modified = embed_coeffs[0], embed_coeffs[1]

    LL1_reconstructed = pywt.idwt2((LL2, (LH2_modified, HL2_modified, HH2)), 'db4', mode='symmetric')
    if LL1_reconstructed.shape != LL1.shape:
        LL1_reconstructed = LL1_reconstructed[:LL1.shape[0], :LL1.shape[1]]

    watermarked_float = pywt.idwt2((LL1_reconstructed, (LH1, HL1, HH1)), 'db4', mode='symmetric')
    if watermarked_float.shape != image_float.shape:
        watermarked_float = watermarked_float[:image_float.shape[0], :image_float.shape[1]]

    # 6. Clip values to the valid 0-255 range and convert back to uint8
    watermarked_float = np.clip(watermarked_float, 0, 255)
    watermarked_image = watermarked_float.astype(np.uint8)

    return watermarked_image


def detection(original_image: np.ndarray, watermarked_image: np.ndarray) -> np.ndarray:
    """
    Function to detect and extract the watermark from an image using DWT.
    This is a non-blind method as it requires the original image.

    :param original_image: The original grayscale image as a NumPy array (uint8).
    :param watermarked_image: The watermarked grayscale image as a NumPy array (uint8).
    :return: The extracted watermark as a 1D NumPy array of bits.
    """
    WATERMARK_SIZE = 1024  # Must be the same size used during embedding
    extracted_watermark = []

    # 1. Convert both images to float for processing
    original_float = original_image.astype(np.float64)
    watermarked_float = watermarked_image.astype(np.float64)

    # 2. Apply a 2-level DWT decomposition to BOTH images
    # Original image coefficients
    coeffs1_orig = pywt.dwt2(original_float, 'db4', mode='symmetric')
    LL1_orig, _ = coeffs1_orig
    coeffs2_orig = pywt.dwt2(LL1_orig, 'db4', mode='symmetric')
    _, (LH2_orig, HL2_orig, _) = coeffs2_orig

    # Watermarked image coefficients
    coeffs1_w = pywt.dwt2(watermarked_float, 'db4', mode='symmetric')
    LL1_w, _ = coeffs1_w
    coeffs2_w = pywt.dwt2(LL1_w, 'db4', mode='symmetric')
    _, (LH2_w, HL2_w, _) = coeffs2_w

    # 3. Prepare the coefficients for extraction
    original_coeffs_to_scan = [LH2_orig, HL2_orig]
    watermarked_coeffs_to_scan = [LH2_w, HL2_w]

    # 4. Extract the watermark by reversing the quantization process
    watermark_idx = 0
    block_size = 8

    for orig_coeff_array, w_coeff_array in zip(original_coeffs_to_scan, watermarked_coeffs_to_scan):
        if watermark_idx >= WATERMARK_SIZE:
            break

        # Iterate over the coefficient arrays in the same block order
        for i in range(0, w_coeff_array.shape[0], block_size):
            for j in range(0, w_coeff_array.shape[1], block_size):
                if watermark_idx >= WATERMARK_SIZE:
                    break

                # Extract the corresponding blocks from both images
                end_i = min(i + block_size, w_coeff_array.shape[0])
                end_j = min(j + block_size, w_coeff_array.shape[1])

                block_orig = orig_coeff_array[i:end_i, j:end_j]
                block_w = w_coeff_array[i:end_i, j:end_j]

                # Flatten the blocks to iterate through coefficients
                block_flat_orig = block_orig.flatten()
                block_flat_w = block_w.flatten()

                for k in range(min(len(block_flat_w), WATERMARK_SIZE - watermark_idx)):
                    original_coeff_val = block_flat_orig[k]
                    watermarked_coeff_val = block_flat_w[k]

                    # Reverse the Quantization Index Modulation (QIM)
                    if abs(original_coeff_val) > 1e-6:
                        # Recalculate the exact same quantization step 'Q'
                        Q = max(0.1, abs(original_coeff_val) * 0.01)

                        # Quantize the watermarked coefficient
                        quantized = np.round(watermarked_coeff_val / Q)

                        # The embedded bit is determined by the parity (even/odd)
                        if int(quantized) % 2 == 0:
                            extracted_bit = 0
                        else:
                            extracted_bit = 1

                        extracted_watermark.append(extracted_bit)
                        watermark_idx += 1

    # Ensure the length is correct
    if len(extracted_watermark) > WATERMARK_SIZE:
        extracted_watermark = extracted_watermark[:WATERMARK_SIZE]

    return np.array(extracted_watermark, dtype=np.uint8)
