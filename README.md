# CTM_ecorp


# embedding_ecorp_2

Overview
This function implements a DWT-based (Discrete Wavelet Transform) watermarking technique that embeds a binary watermark into an image using frequency domain manipulation. It's specifically optimized for high WPSNR (Weighted Peak Signal-to-Noise Ratio) while maintaining robustness.

Theoretical Background
DWT Decomposition
The DWT transforms an image from spatial domain to frequency domain, decomposing it into:

LL: Low-Low (approximation coefficients - contains main image content)
LH: Low-High (horizontal detail coefficients)
HL: High-Low (vertical detail coefficients)
HH: High-High (diagonal detail coefficients)
Step-by-Step Breakdown

1. Initialization and Loading
 FIXED_SEED = 42
WATERMARK_SIZE = 1024
np.random.seed(FIXED_SEED)

-Sets deterministic random seed for reproducible results
-Defines watermark size as 1024 bits
-Loads grayscale image and binary watermark data

2. Three-Level DWT Decomposition
# Level 1
coeffs1 = pywt.dwt2(image_float, 'db4', mode='symmetric')
LL1, (LH1, HL1, HH1) = coeffs1

# Level 2 - decomposes LL1 further
coeffs2 = pywt.dwt2(LL1, 'db4', mode='symmetric')
LL2, (LH2, HL2, HH2) = coeffs2

# Level 3 - decomposes LL2 further
coeffs3 = pywt.dwt2(LL2, 'db4', mode='symmetric')
LL3, (LH3, HL3, HH3) = coeffs3

Theoretical Operation:

-Uses Daubechies-4 ('db4') wavelet with symmetric boundary conditions
-Each level creates a pyramid structure where:
    Level 1: Full resolution decomposition
    Level 2: Half resolution (from LL1)
    Level 3: Quarter resolution (from LL2)

3. Adaptive Embedding Strength Calculation

def calculate_adaptive_strength(coeff_block, base_strength=0.001):
    local_variance = np.var(coeff_block)
    if local_variance > 100:
        return base_strength * 2.0
    elif local_variance > 50:
        return base_strength * 1.5
    # ... etc

Purpose:

-Adapts embedding strength based on local image characteristics
-Higher variance areas (edges, textures) can tolerate stronger modifications
-Lower variance areas (smooth regions) need gentler modifications
-Very small base strength (0.001) ensures high WPSNR

4. Target Coefficient Selection

embed_coeffs = [LH2.copy(), HL2.copy()]

Strategic Choice:

Embeds in Level 2 detail coefficients (LH2, HL2)

Why Level 2? Balance between:
-Robustness: Not too high-frequency (survives compression)
-Imperceptibility: Not too low-frequency (avoids affecting main content)

Why LH2/HL2? Mid-frequency detail coefficients are less perceptually important

5. Block-Based Embedding

block_size = 8
for i in range(0, array_2d.shape[0], block_size):
    for j in range(0, array_2d.shape[1], block_size):

Theoretical Advantage:
-Processes 8×8 blocks for local adaptation
-Each block gets its own embedding strength
-Mimics human visual system's local processing


6. Quantization-Based Embedding Method

Q = max(0.1, abs(coeff_val) * 0.01)
quantized = np.round(coeff_val / Q)

if bit == 1:
    # Make quantized value odd
    if int(quantized) % 2 == 0:
        quantized += 1 if coeff_val >= 0 else -1
else:
    # Make quantized value even
    if int(quantized) % 2 == 1:
        quantized += 1 if coeff_val >= 0 else -1

Theoretical Operation:

-Quantization Step (Q): Proportional to coefficient magnitude (1% of value)
-Bit Embedding: Modifies the parity (odd/even) of quantized values
    Bit 1 → Force quantized value to be odd
    Bit 0 → Force quantized value to be even
-Modification: new_val = quantized * Q
-Final Update: coeff_val + (new_val - coeff_val) * alpha

7. Reconstruction Process

# Level 3 → Level 2
LL2_reconstructed = pywt.idwt2((LL3, (LH3, HL3, HH3)), 'db4')

# Level 2 → Level 1 (with modified LH2, HL2)
LL1_reconstructed = pywt.idwt2((LL2_reconstructed, (LH2_modified, HL2_modified, HH2)), 'db4')

# Level 1 → Final image
watermarked_float = pywt.idwt2((LL1_reconstructed, (LH1, HL1, HH1)), 'db4')

Size Handling:

DWT/IDWT can cause small size mismatches
Code includes careful cropping/padding to maintain original dimensions
Key Theoretical Advantages
Frequency Domain Robustness: Watermark survives compression and filtering
Adaptive Strength: Preserves image quality in smooth areas
Mid-Frequency Embedding: Balance between robustness and imperceptibility
Quantization-Based: More robust than simple additive methods
Multi-Level: Uses wavelet pyramid for hierarchical embedding
Quality vs. Robustness Trade-off
High WPSNR (>66 dB): Achieved through very small embedding strength (0.001)
Robustness: Maintained through frequency domain embedding and quantization
Perceptual Quality: Protected by adaptive strength and mid-frequency targeting
This implementation represents a sophisticated approach to digital watermarking that leverages wavelet theory for optimal embedding in the frequency domain.