import cv2
import numpy as np
from scipy.signal import convolve2d
import pywt
import os


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


def extraction(input1, input2): # 1 original, 2 watermarked
    
    """
    Non-blind watermark extraction
    input1: string of watermarked image filename (BMP format)
    input2: string of original image filename (BMP format)
    returns: 1024-bit watermark as numpy array
    """
    
    # Constants (have to correspond to embedding)
    FIXED_SEED = 42
    mask = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]
    rng = np.random.default_rng(FIXED_SEED)
    PN0 = rng.standard_normal(len(mask))
    PN1 = rng.standard_normal(len(mask))
    
    # Read original image
    Io = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if Io is None:
        raise ValueError(f"Cannot read original image: {input1}")
    Io = Io.astype(np.float32)
    
    # Read watermarked image
    Iw = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    if Iw is None:
        raise ValueError(f"Cannot read watermarked image: {input2}")
    Iw = Iw.astype(np.float32)
    
    # DWT decomposition of BOTH images
    coeffs_w = pywt.wavedec2(Iw, 'db2', level=3)
    coeffs_o = pywt.wavedec2(Io, 'db2', level=3)
    
    # Extract subbands from watermarked image
    (LH3_w, HL3_w, HH3_w), (LH2_w, HL2_w, HH2_w), (LH1_w, HL1_w, HH1_w) = coeffs_w[1], coeffs_w[2], coeffs_w[3]
    bands_w = [HL3_w, LH3_w, HL2_w, LH2_w]
    
    # Extract subbands from original image
    (LH3_o, HL3_o, HH3_o), (LH2_o, HL2_o, HH2_o), (LH1_o, HL1_o, HH1_o) = coeffs_o[1], coeffs_o[2], coeffs_o[3]
    bands_o = [HL3_o, LH3_o, HL2_o, LH2_o]

    # Extract 1024 bits
    bits = np.zeros(1024, dtype=np.uint8)
    idx = 0
    
    for b in range(4):
        Bw = bands_w[b]  # watermarked band
        Bo = bands_o[b]  # original band
        
        for by in range(0, 64, 4):
            for bx in range(0, 64, 4):
                # DCT of watermarked block
                Cw = cv2.dct(Bw[by:by+4, bx:bx+4])
                
                # DCT of original block
                Co = cv2.dct(Bo[by:by+4, bx:bx+4])
                
                # DIFFERENCE: isolate the watermark signal
                # The embedded watermark is: Cw ≈ Co + alpha * PN
                # So: (Cw - Co) ≈ alpha * PN
                Cdiff = Cw - Co
                
                # Extract mid-frequency coefficients from DIFFERENCE
                x = np.array([Cdiff[u,v] for (u,v) in mask], dtype=np.float32)
                
                # Correlation with PN sequences
                # Now x should be closer to alpha*PN0 or alpha*PN1
                rho0 = (x @ PN0) / (np.linalg.norm(x) * np.linalg.norm(PN0) + 1e-12)
                rho1 = (x @ PN1) / (np.linalg.norm(x) * np.linalg.norm(PN1) + 1e-12)
                
                # Bit decision
                bits[idx] = 1 if rho1 > rho0 else 0
                idx += 1
    
    return bits


def detection(input1, input2, input3):
    """
    Wrapper di detection non-blind che:
    - input1: path dell'originale (string)
    - input2: path dell'immagine watermarked (string)
    - input3: path dell'immagine attaccata (string)
    Ritorna:
      (present_flag, wpsnr_value)
      where present_flag = 1 se watermark presente (similarity >= tau), 0 altrimenti.
    """
    tau = 0.517647  # la tua soglia (mantienila o sostituiscila con la tua tau calcolata)

    # VALIDAZIONE PATH: detection deve leggere le immagini internamente ma possiamo anche usarle qui per WPSNR
    if not isinstance(input1, (str, bytes, os.PathLike)):
        raise TypeError("detection: input1 deve essere un path (string).")
    if not isinstance(input2, (str, bytes, os.PathLike)):
        raise TypeError("detection: input2 deve essere un path (string).")
    if not isinstance(input3, (str, bytes, os.PathLike)):
        raise TypeError("detection: input3 deve essere un path (string).")

    # Leggi immagini solo per calcolo WPSNR (extraction farà i suoi imread internamente)
    I_orig = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    I_w    = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    I_att  = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)

    if I_orig is None:
        raise ValueError(f"Cannot read original image at path: {input1}")
    if I_w is None:
        raise ValueError(f"Cannot read watermarked image at path: {input2}")
    if I_att is None:
        raise ValueError(f"Cannot read attacked image at path: {input3}")

    # 1) Extractions: PASSIAMO I PATH (stringhe) a extraction() come previsto
    watermark_extracted = extraction(input1, input2)  # original_path, watermarked_path
    watermark_attacked   = extraction(input1, input3)  # original_path, attacked_path

    # 2) Similarity / Hamming
    sim = similarity(watermark_extracted, watermark_attacked)
    hd = int(np.sum(np.abs(watermark_extracted.astype(np.int32) - watermark_attacked.astype(np.int32))))

    output1 = 1 if sim >= tau else 0  # present/not present

    # 3) WPSNR tra watermarked e attacked (qui usiamo gli array letti sopra)
    output2 = float(wpsnr(I_w, I_att))

    return output1, output2

