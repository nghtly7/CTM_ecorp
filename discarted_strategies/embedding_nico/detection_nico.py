import cv2
import numpy as np
from scipy.signal import convolve2d
import pywt


def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s


# ---------- CSF / WPSNR (unchanged) ----------
def _csffun(u, v):
    f = np.sqrt(u**2 + v**2)
    w = 2 * np.pi * f / 60
    sigma = 2
    Sw = 1.5 * np.exp(-sigma**2 * w**2 / 2) - np.exp(-2 * sigma**2 * w**2 / 2)
    sita = np.arctan2(v, u)
    bita = 8
    f0 = 11.13
    w0 = 2 * np.pi * f0 / 60
    exp_term = np.exp(bita * (w - w0))
    Ow = (1 + exp_term * (np.cos(2 * sita))**4) / (1 + exp_term)
    return Sw * Ow

def _csfmat():
    fr = np.arange(-20, 21, 1)
    u, v = np.meshgrid(fr, fr, indexing='xy')
    return _csffun(u, v)

def _get_csf_filter():
    Fmat = _csfmat()
    fc = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Fmat)))
    return np.real(fc)

def wpsnr(A, B):
    if A.dtype != np.float32 and A.dtype != np.float64:
        A = A.astype(np.float64) / 255.0
    if B.dtype != np.float32 and B.dtype != np.float64:
        B = B.astype(np.float64) / 255.0
    if np.array_equal(A, B):
        return 9999999.0
    e = A - B
    fc = _get_csf_filter()
    we = convolve2d(e, fc, mode='same', boundary='wrap')
    wmse = np.mean(we**2)
    if wmse == 0:
        return 9999999.0
    return 20 * np.log10(1.0 / np.sqrt(wmse))


# ---------- Helper: block selection ----------
def top_variance_positions(B, block=4, nblocks=150):
    h, w = B.shape
    h_eff, w_eff = min(64, h), min(64, w)
    positions, scores = [], []
    for by in range(0, h_eff, block):
        for bx in range(0, w_eff, block):
            blk = B[by:by+block, bx:bx+block]
            positions.append((by, bx))
            scores.append(np.var(blk))
    idx = np.argsort(scores)[::-1][:nblocks]
    return [positions[i] for i in idx]


# ---------- Extraction ----------
def extraction(original_path, watermarked_path):
    """
    Non-blind extraction matching the updated embedding.
    """
    FIXED_SEED = 42
    TOP_BLOCKS = 150

    # same masks and PN as embedding v3
    mask_L3 = [(1,2), (2,1), (2,2), (1,3), (3,1)]
    mask_L2 = [(1,0), (0,1), (1,1), (2,1), (1,2)]

    rng = np.random.default_rng(FIXED_SEED)
    PN0_L3 = rng.standard_normal(len(mask_L3))
    PN1_L3 = rng.standard_normal(len(mask_L3))
    PN0_L2 = rng.standard_normal(len(mask_L2))
    PN1_L2 = rng.standard_normal(len(mask_L2))

    # read images
    Io = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    Iw = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # DWT decomposition
    coeffs_o = pywt.wavedec2(Io, 'db2', level=3)
    coeffs_w = pywt.wavedec2(Iw, 'db2', level=3)

    (LH3_o, HL3_o, HH3_o), (LH2_o, HL2_o, HH2_o), (LH1_o, HL1_o, HH1_o) = coeffs_o[1], coeffs_o[2], coeffs_o[3]
    (LH3_w, HL3_w, HH3_w), (LH2_w, HL2_w, HH2_w), (LH1_w, HL1_w, HH1_w) = coeffs_w[1], coeffs_w[2], coeffs_w[3]

    bands_o = [HL3_o, LH3_o, HL2_o, LH2_o]
    bands_w = [HL3_w, LH3_w, HL2_w, LH2_w]

    bits = []
    for b in range(4):
        Bo, Bw = bands_o[b], bands_w[b]
        pos = top_variance_positions(Bo, 4, TOP_BLOCKS)
        if b <= 1:
            mask, PN0, PN1 = mask_L3, PN0_L3, PN1_L3
        else:
            mask, PN0, PN1 = mask_L2, PN0_L2, PN1_L2
        for (by, bx) in pos:
            Cw = cv2.dct(Bw[by:by+4, bx:bx+4])
            Co = cv2.dct(Bo[by:by+4, bx:bx+4])
            Cdiff = Cw - Co
            x = np.array([Cdiff[u, v] for (u, v) in mask], dtype=np.float32)
            rho0 = (x @ PN0) / (np.linalg.norm(x) * np.linalg.norm(PN0) + 1e-12)
            rho1 = (x @ PN1) / (np.linalg.norm(x) * np.linalg.norm(PN1) + 1e-12)
            bits.append(1 if rho1 > rho0 else 0)
    return np.array(bits[:1024], dtype=np.uint8)


# ---------- Detection ----------
def detection(input1, input2, input3):
    """
    input1: original
    input2: watermarked
    input3: attacked
    """
    tau = 0.52  # threshold (slightly tuned)
    I_orig = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    I_w = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    I_att = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)

    wm_extracted = extraction(input1, input2)
    wm_attacked = extraction(input1, input3)

    sim = similarity(wm_extracted, wm_attacked)
    hd = np.sum(np.abs(wm_extracted - wm_attacked))

    detected = 1 if sim >= tau else 0
    wps = wpsnr(I_w, I_att)

    return detected, float(wps)
