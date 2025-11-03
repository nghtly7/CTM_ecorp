import cv2
import numpy as np
from scipy.signal import convolve2d
import pywt
import time

FIXED_SEED = 42
WAVELET = "db2"
LEVELS = 3
MASK_IDX = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]
EPS = 1e-8
# TAU = 0.525185   # decision threshold
TAU = 0.536124   # decision threshold


MIN_DETECTION_TIME = 4.0

_rng = np.random.default_rng(FIXED_SEED)
_PN0 = _rng.standard_normal(len(MASK_IDX)).astype(np.float32)
_PN1 = _rng.standard_normal(len(MASK_IDX)).astype(np.float32)
_PN0 = _PN0 / (np.linalg.norm(_PN0) + EPS)
_PN1 = _PN1 - (_PN0 @ _PN1) * _PN0
_PN1 = _PN1 / (np.linalg.norm(_PN1) + EPS)
# _DELTA_PN = (_PN1 - _PN0).astype(np.float32)
# _DELTA_PN_NORM = _DELTA_PN / (np.linalg.norm(_DELTA_PN) + EPS)

#! --- Allineamento ad embedding: HVS sulle PN ---
HVS_WEIGHTS = np.array([0.55, 0.60, 0.95, 1.05, 1.05, 1.15, 1.15], dtype=np.float32)

_PN0_H = _PN0 * HVS_WEIGHTS
_PN1_H = _PN1 * HVS_WEIGHTS

_DELTA_PN = (_PN1_H - _PN0_H).astype(np.float32)
_DELTA_PN_NORM = _DELTA_PN / (np.linalg.norm(_DELTA_PN) + EPS)


def _load_image_gray(path: str) -> np.ndarray:
    """Load grayscale image, raise descriptive error if not found."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"detection_stronger_v2: impossibile leggere l'immagine: {path}")
    return img.astype(np.float32)

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def _csffun(u, v):
    """
    Computes the Contrast Sensitivity Function (CSF) value for given spatial frequencies.
    This is a Python translation of csffun.m.
    """
    # Calculate radial frequency
    f = np.sqrt(u**2 + v**2)
    w = 2 * np.pi * f / 60

    # Intermediate spatial frequency response
    sigma = 2
    Sw = 1.5 * np.exp(-sigma**2 * w**2 / 2) - np.exp(-2 * sigma**2 * w**2 / 2)

    # High-frequency modification
    sita = np.arctan2(v, u) # Use arctan2 for quadrant correctness
    bita = 8
    f0 = 11.13
    w0 = 2 * np.pi * f0 / 60
    
    # Avoid division by zero or overflow in exp
    exp_term = np.exp(bita * (w - w0))
    Ow = (1 + exp_term * (np.cos(2 * sita))**4) / (1 + exp_term)
    
    # Final response
    Sa = Sw * Ow
    return Sa


def _csfmat():
    """
    Computes the CSF frequency response matrix.
    This is a Python translation of csfmat.m.
    """
    # Define frequency range
    min_f, max_f, step_f = -20, 20, 1
    freq_range = np.arange(min_f, max_f + step_f, step_f)
    n = len(freq_range)
    
    # Create frequency grids
    u, v = np.meshgrid(freq_range, freq_range, indexing='xy')
    
    # Compute the frequency response matrix by calling _csffun
    Fmat = _csffun(u, v)
    
    return Fmat


def _get_csf_filter():
    """
    Computes the 2D filter coefficients for the CSF.
    This is a Python translation of csf.m, which uses fsamp2.
    The fsamp2 function is implemented using an inverse Fourier transform.
    """
    # 1. Get the frequency response matrix
    Fmat = _csfmat()
    
    # 2. Compute the 2D filter coefficients using the frequency sampling method
    # This is equivalent to MATLAB's fsamp2(Fmat)
    # The shifts are necessary to handle the centered frequency response
    filter_coeffs = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Fmat)))
    
    # The filter coefficients should be real
    return np.real(filter_coeffs)


def wpsnr(image_a, image_b):
    """
    Computes the Weighted Peak Signal-to-Noise Ratio (WPSNR) between two images.

    This function is a Python translation of the provided WPSNR.m script. It uses a 
    Contrast Sensitivity Function (CSF) to weigh the spatial frequencies of the error image.

    Args:
        image_a (np.ndarray):   The original image, as a NumPy array. 
                                Values can be uint8 (0-255) or float (0.0-1.0).
        image_b (np.ndarray):   The distorted image, as a NumPy array. 
                                Must have the same dimensions and type as image_a.

    Returns:
        float: The WPSNR value in decibels (dB).
    """
    # --- Data validation and normalization ---
    if not isinstance(image_a, np.ndarray) or not isinstance(image_b, np.ndarray):
        raise TypeError("Input images must be NumPy arrays.")

    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Normalize images to the [0, 1] range if they are not already floats
    if image_a.dtype != np.float64 and image_a.dtype != np.float32:
        A = image_a.astype(np.float64) / 255.0
    else:
        A = image_a.copy()

    if image_b.dtype != np.float64 and image_b.dtype != np.float32:
        B = image_b.astype(np.float64) / 255.0
    else:
        B = image_b.copy()
        
    if A.max() > 1.0 or A.min() < 0.0 or B.max() > 1.0 or B.min() < 0.0:
        raise ValueError("Input image values must be in the interval [0, 1] for floats or [0, 255] for integers.")

    # --- WPSNR Calculation ---
    # Handle identical images case
    if np.array_equal(A, B):
        return 9999999.0  # Return a large number for infinite PSNR, as in the original code

    # 1. Calculate the error image
    error_image = A - B
    
    # 2. Get the Contrast Sensitivity Function (CSF) filter
    csf_filter = _get_csf_filter()
    
    # 3. Filter the error image with the CSF filter (2D convolution)
    # This is equivalent to MATLAB's filter2(fc, e)
    weighted_error = convolve2d(error_image, csf_filter, mode='same', boundary='wrap')
    
    # 4. Calculate the weighted mean squared error (WMSE)
    wmse = np.mean(weighted_error**2)
    
    # 5. Calculate WPSNR
    # The peak signal value is 1.0 because the images are normalized
    if wmse == 0:
        return 9999999.0 # Should be caught by the identity check, but included for safety
    
    decibels = 20 * np.log10(1.0 / np.sqrt(wmse))
    
    return decibels

def extraction(original_path: str, test_path: str) -> np.ndarray:
    """
    Extract 1024 bits from test_path relative to original_path following
    the embedding_stronger_v2 scheme.

    Returns:
        bits: np.ndarray shape (1024,), dtype=np.uint8 with values {0,1}
    """
    # read images
    I_orig = _load_image_gray(original_path)
    I_test = _load_image_gray(test_path)

    # DWT decomposition
    coeffs_o = pywt.wavedec2(I_orig, wavelet=WAVELET, level=LEVELS)
    coeffs_t = pywt.wavedec2(I_test, wavelet=WAVELET, level=LEVELS)

    # extract the same 4 subbands used in embedding (HL3, LH3, HL2, LH2)
    (LH3_o, HL3_o, HH3_o), (LH2_o, HL2_o, HH2_o), (LH1_o, HL1_o, HH1_o) = coeffs_o[1], coeffs_o[2], coeffs_o[3]
    (LH3_t, HL3_t, HH3_t), (LH2_t, HL2_t, HH2_t), (LH1_t, HL1_t, HH1_t) = coeffs_t[1], coeffs_t[2], coeffs_t[3]

    bands_orig = [("HL", 3, HL3_o), ("LH", 3, LH3_o), ("HL", 2, HL2_o), ("LH", 2, LH2_o)]
    bands_test = [("HL", 3, HL3_t), ("LH", 3, LH3_t), ("HL", 2, HL2_t), ("LH", 2, LH2_t)]

    scores = []  # projection scores per block (float)
    bits = []    # decided bits per block

    # iterate blocks in the exact same order as embedding
    for b_idx in range(4):
        B_o = bands_orig[b_idx][2]
        B_t = bands_test[b_idx][2]
        # B_* expected to be 64x64 (level-specific)
        for by in range(0, 64, 4):
            for bx in range(0, 64, 4):
                # get 4x4 blocks and DCT
                block_o = B_o[by:by+4, bx:bx+4].astype(np.float32, copy=True)
                block_t = B_t[by:by+4, bx:bx+4].astype(np.float32, copy=True)

                Co = cv2.dct(block_o)
                Ct = cv2.dct(block_t)

                # mid-band vectors
                mb_o = np.array([Co[u, v] for (u, v) in MASK_IDX], dtype=np.float32)
                mb_t = np.array([Ct[u, v] for (u, v) in MASK_IDX], dtype=np.float32)

                # delta and normalization by block std (as embedding equalization)
                delta = mb_t - mb_o
                block_sigma = float(np.std(mb_o)) + EPS
                delta_n = delta / block_sigma

                # project onto normalized DELTA_PN
                score = float(np.dot(delta_n, _DELTA_PN_NORM))
                scores.append(score)

                # decision bit via sign
                bit = 1 if score > 0.0 else 0
                bits.append(bit)

    # ensure length == 1024 (embedding expects 1024 blocks)
    bits_arr = np.asarray(bits, dtype=np.uint8)
    if bits_arr.size != 1024:
        # if mismatch, we try to pad or truncate to 1024 preserving order
        if bits_arr.size < 1024:
            pad_size = 1024 - bits_arr.size
            bits_arr = np.concatenate([bits_arr, np.zeros(pad_size, dtype=np.uint8)])
        else:
            bits_arr = bits_arr[:1024]

    return bits_arr


def detection(original_path: str, wm_path: str, attacked_path: str) -> tuple:
    """
    Non-blind detection wrapper that uses extract_bits().

    Returns:
        (present_flag, wpsnr_value)
        - present_flag: int (1 if watermark present, 0 otherwise)
        - wpsnr_value: float (wpsnr(wm, attacked) or PSNR fallback)
    """
    t0 = time.time() 
    # extract bits from watermarked and attacked (relative to original)
    bits_w = extraction(original_path, wm_path)
    bits_a = extraction(original_path, attacked_path)

    # compute similarity (1 - normalized Hamming distance)
    sim = similarity(bits_w.astype(np.float32), bits_a.astype(np.float32))


    # compute wpsnr between watermarked and attacked (for logging / thresholds)
    I_w = _load_image_gray(wm_path)
    I_a = _load_image_gray(attacked_path)
    # passiamo uint8 puliti a wpsnr (evita l'errore di range)
    I_w_u8 = np.clip(I_w, 0, 255).astype(np.uint8)
    I_a_u8 = np.clip(I_a, 0, 255).astype(np.uint8)
    wpsnr_val = wpsnr(I_w_u8, I_a_u8)

    present_flag = 1 if sim >= TAU else 0
    
    elapsed = time.time() - t0
    if elapsed < MIN_DETECTION_TIME:
        time.sleep(MIN_DETECTION_TIME - elapsed)

    return int(present_flag), float(wpsnr_val)