import cv2
import numpy as np
import pywt
from wpsnr import wpsnr as WPSNR  # richiesto dal regolamento

# --- Costanti GLOBALI (devono coincidere con l'embedding!) ---
FIXED_SEED = 42
mask = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]
rng = np.random.default_rng(FIXED_SEED)
PN0 = rng.standard_normal(len(mask))
PN1 = rng.standard_normal(len(mask))

# threshold iniziale (poi si affina con ROC)
tau_global = 0.75


def extract_bits(Iimg):
    Iimg = Iimg.astype(np.float32)

    coeffs = pywt.wavedec2(Iimg, 'db2', level=3)
    (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs[1], coeffs[2], coeffs[3]
    bands = [HL3, LH3, HL2, LH2]

    bits = np.zeros(1024, dtype=np.uint8)
    idx = 0
    for b in range(4):
        B = bands[b]
        for by in range(0,64,4):
            for bx in range(0,64,4):
                C = cv2.dct(B[by:by+4, bx:bx+4])
                x = np.array([C[u,v] for (u,v) in mask], dtype=np.float32)
                rho0 = (x @ PN0) / (np.linalg.norm(x)*np.linalg.norm(PN0) + 1e-12)
                rho1 = (x @ PN1) / (np.linalg.norm(x)*np.linalg.norm(PN1) + 1e-12)
                bits[idx] = 1 if rho1 > rho0 else 0
                idx += 1
    return bits


def detection(input1, input2, input3):
    I_orig = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    I_w    = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    I_att  = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)

    # 1) Estrai i bit
    bits_w   = extract_bits(I_w)
    bits_att = extract_bits(I_att)

    # 2) Similarità e decisione
    hd = np.count_nonzero(bits_w ^ bits_att)
    sim = 1.0 - hd/1024.0
    present = 1 if sim >= tau_global else 0

    # 3) WPSNR tra watermarked e attacked
    wpsnr_val = WPSNR(I_w, I_att)

    return present, float(wpsnr_val)
