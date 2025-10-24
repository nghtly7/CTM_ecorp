import cv2
import numpy as np
import pywt

def extraction(input1, input2):
    """
    Non-blind watermark extraction
    input1: string of watermarked image filename (BMP format)
    input2: string of original image filename (BMP format)
    returns: 1024-bit watermark as numpy array
    """
    
    # --- Costanti (devono coincidere con l'embedding!) ---
    FIXED_SEED = 42
    mask = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]
    rng = np.random.default_rng(FIXED_SEED)
    PN0 = rng.standard_normal(len(mask))
    PN1 = rng.standard_normal(len(mask))
    
    # Read watermarked image
    Iw = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if Iw is None:
        raise ValueError(f"Cannot read watermarked image: {input1}")
    Iw = Iw.astype(np.float32)
    
    # Read original image
    Io = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    if Io is None:
        raise ValueError(f"Cannot read original image: {input2}")
    Io = Io.astype(np.float32)

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



