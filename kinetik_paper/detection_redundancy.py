from wpsnr import wpsnr
import numpy as np
import cv2
import pywt

TAU = 0.80
ALPHA = 0.2  # deve combaciare con l'embedding C

def similarity(X, X_star):
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def _dct2(img):
    return cv2.dct(img.astype(np.float32))

def _dwt2_levels(img, wavelet='haar'):
    cA1, (cH1, cV1, cD1) = pywt.dwt2(img, wavelet)
    cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, wavelet)
    return cA2, (cH2, cV2, cD2), (cA1, (cH1, cV1, cD1))

def _estimate_bipolar_from_combined(E):
    """
    Majority voting C1 sui bit:
    - p1: identità → sign(E)
    - p2: inverse of row-roll(+1) → sign(roll(E, -1, axis=0))
    - p3: inverse of col-roll(+1) → sign(roll(E, -1, axis=1))
    voto per cella: segno della somma; 0 viene trattato come +1.
    Ritorna una matrice bipolare in {-1,+1}.
    """
    s1 = np.sign(E)
    s2 = np.sign(np.roll(E, shift=-1, axis=0))
    s3 = np.sign(np.roll(E, shift=-1, axis=1))
    # sostituisco gli zeri con +1 per evitare indecisioni
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    s3[s3 == 0] = 1
    vote = np.sign(s1 + s2 + s3)
    vote[vote == 0] = 1
    return vote.astype(np.float32)

def _extract_bipolar_LL2_spread(LL2_like, LL2_host, alpha, sigma_host):
    """E = (LL2_like - LL2_host) / (alpha * sigma_host)  → majority voting su tre allineamenti."""
    E = (LL2_like[:32, :32] - LL2_host[:32, :32]) / (alpha * (sigma_host + 1e-12))
    return _estimate_bipolar_from_combined(E)  # (32,32) in {-1,+1}

def detection(input1, input2, input3):
    host = cv2.imread(input1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    wm   = cv2.imread(input2, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    att  = cv2.imread(input3, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if host is None or wm is None or att is None or host.shape != (512,512) or wm.shape != (512,512) or att.shape != (512,512):
        raise ValueError("All images must be readable 512x512 grayscale.")

    Hd = _dct2(host); Wd = _dct2(wm); Ad = _dct2(att)

    H_LL2, _, _ = _dwt2_levels(Hd)
    W_LL2, _, _ = _dwt2_levels(Wd)
    A_LL2, _, _ = _dwt2_levels(Ad)

    sigma_host = float(H_LL2.std())

    # estraggo direttamente matrici bipolari {-1,+1} tramite majority voting
    X_bip  = _extract_bipolar_LL2_spread(W_LL2, H_LL2, ALPHA, sigma_host).ravel()
    Xs_bip = _extract_bipolar_LL2_spread(A_LL2, H_LL2, ALPHA, sigma_host).ravel()

    sim = float(similarity(X_bip, Xs_bip))
    output1 = 1 if sim >= TAU else 0
    output2 = float(wpsnr(wm.astype(np.uint8), att.astype(np.uint8)))
    return output1, output2
