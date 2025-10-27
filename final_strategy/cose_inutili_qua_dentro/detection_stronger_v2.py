import cv2
import numpy as np
from scipy.signal import convolve2d
import pywt

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


def detection(input1, input2, input3):
    """
    detection(non-blind) compatibile con embedding_stronger_v2.py

    input1: path originale (string)
    input2: path watermarked (string)
    input3: path attacked (string)

    ritorna: (present_flag, wpsnr_value)
      - present_flag: 1 se similarity >= tau, 0 altrimenti
      - wpsnr_value: valore calcolato usando la funzione wpsnr del tuo modulo (se disponibile)
    """
    # --------- parametri (mantieni sincronizzati con embedding) ----------
    FIXED_SEED = 42
    WAVELET = "db2"
    LEVELS = 3
    MASK_IDX = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]
    EPS = 1e-8
    TAU = 0.517647   # soglia decisione, mantieni la tua precedente o ricalibra via ROC

    I_orig = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    I_w    = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    I_att  = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)

    if I_orig is None:
        raise FileNotFoundError(f"detection: impossibile leggere {input1}")
    if I_w is None:
        raise FileNotFoundError(f"detection: impossibile leggere {input2}")
    if I_att is None:
        raise FileNotFoundError(f"detection: impossibile leggere {input3}")

    I_orig_f = I_orig.astype(np.float32)
    I_w_f    = I_w.astype(np.float32)
    I_att_f  = I_att.astype(np.float32)

    # --------- calcola DWT dei tre (usiamo pywt) ----------
    coeffs_orig = pywt.wavedec2(I_orig_f, wavelet=WAVELET, level=LEVELS)
    coeffs_w    = pywt.wavedec2(I_w_f,    wavelet=WAVELET, level=LEVELS)
    coeffs_att  = pywt.wavedec2(I_att_f,  wavelet=WAVELET, level=LEVELS)

    # estrai le 4 bande usate (HL3, LH3, HL2, LH2)
    (LH3_o, HL3_o, HH3_o), (LH2_o, HL2_o, HH2_o), (LH1_o, HL1_o, HH1_o) = coeffs_orig[1], coeffs_orig[2], coeffs_orig[3]
    (LH3_w, HL3_w, HH3_w), (LH2_w, HL2_w, HH2_w), (LH1_w, HL1_w, HH1_w) = coeffs_w[1], coeffs_w[2], coeffs_w[3]
    (LH3_a, HL3_a, HH3_a), (LH2_a, HL2_a, HH2_a), (LH1_a, HL1_a, HH1_a) = coeffs_att[1], coeffs_att[2], coeffs_att[3]

    bands_orig = [("HL",3,HL3_o), ("LH",3,LH3_o), ("HL",2,HL2_o), ("LH",2,LH2_o)]
    bands_w    = [("HL",3,HL3_w), ("LH",3,LH3_w), ("HL",2,HL2_w), ("LH",2,LH2_w)]
    bands_a    = [("HL",3,HL3_a), ("LH",3,LH3_a), ("HL",2,HL2_a), ("LH",2,LH2_a)]

    # --------- genera PN0/PN1 ortonormalizzate (same seed embedding) ----------
    rng = np.random.default_rng(FIXED_SEED)
    PN0 = rng.standard_normal(len(MASK_IDX)).astype(np.float32)
    PN1 = rng.standard_normal(len(MASK_IDX)).astype(np.float32)
    PN0 = PN0 / (np.linalg.norm(PN0) + EPS)
    PN1 = PN1 - (PN0 @ PN1) * PN0
    PN1 = PN1 / (np.linalg.norm(PN1) + EPS)
    DELTA_PN = (PN1 - PN0)
    DELTA_PN_NORM = DELTA_PN / (np.linalg.norm(DELTA_PN) + EPS)

    # --------- scorri blocchi nello stesso ordine usato in embedding e calcola proiezioni ----------
    num_blocks = 0
    # we'll collect per-block scores for watermarked and attacked
    scores_w = []
    scores_a = []

    for b_idx in range(4):  # 4 bands
        # pick arrays in parallel
        B_o = bands_orig[b_idx][2]
        B_w = bands_w[b_idx][2]
        B_a = bands_a[b_idx][2]
        # B_* are 64x64
        for by in range(0, 64, 4):
            for bx in range(0, 64, 4):
                Co_block = B_o[by:by+4, bx:bx+4].astype(np.float32, copy=True)
                Cw_block = B_w[by:by+4, bx:bx+4].astype(np.float32, copy=True)
                Ca_block = B_a[by:by+4, bx:bx+4].astype(np.float32, copy=True)

                # DCT on each block (embedding embedded in DCT domain)
                Co = cv2.dct(Co_block)
                Cw = cv2.dct(Cw_block)
                Ca = cv2.dct(Ca_block)

                # build mid-band vectors
                mb_o = np.array([Co[u,v] for (u,v) in MASK_IDX], dtype=np.float32)
                mb_w = np.array([Cw[u,v] for (u,v) in MASK_IDX], dtype=np.float32)
                mb_a = np.array([Ca[u,v] for (u,v) in MASK_IDX], dtype=np.float32)

                # delta vectors
                delta_w = mb_w - mb_o
                delta_a = mb_a - mb_o

                # block energy normalization (stabilize across blocks)
                block_sigma = np.std(mb_o) + EPS

                # normalize deltas (this mirrors equalizzazione in embedding)
                delta_w_n = delta_w / block_sigma
                delta_a_n = delta_a / block_sigma

                # project onto delta PN (soft evidence)
                score_w = float(np.dot(delta_w_n, DELTA_PN_NORM))   # scalar (can be negative)
                score_a = float(np.dot(delta_a_n, DELTA_PN_NORM))

                scores_w.append(score_w)
                scores_a.append(score_a)

                num_blocks += 1

    # num_blocks should be 1024
    if num_blocks != 1024:
        # something is off — continue but warn
        # raise ValueError(f"detection: expected 1024 blocks, found {num_blocks}")
        pass

    # --------- estimate bits via sign of projection; also provide confidence (abs(score)) ----------
    bits_orig = np.array([1 if s > 0 else 0 for s in scores_w], dtype=np.uint8)  # for reference
    bits_w = np.array([1 if s > 0 else 0 for s in scores_w], dtype=np.uint8)
    bits_a = np.array([1 if s > 0 else 0 for s in scores_a], dtype=np.uint8)

    # similarity: fraction of equal bits (1 - normalized Hamming distance)
    similarity = float(np.sum(bits_w == bits_a)) / float(len(bits_w))

    # compute wpsnr between watermarked and attacked (for logging/criteria)
    wpsnr_val_original = wpsnr(I_orig, I_att)
    wpsnr_val = wpsnr(I_w, I_att)

    # decide presence
    present_flag = 1 if similarity >= TAU else 0

    # return (present_flag, wpsnr)
    return int(present_flag), float(wpsnr_val)
