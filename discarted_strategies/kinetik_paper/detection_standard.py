from wpsnr import wpsnr
import numpy as np
import cv2
import pywt

# soglia placeholder: sostituiscila con quella trovata via ROC
TAU = 0.80
ALPHA = 0.2  # deve combaciare con l'embedding

def similarity(X, X_star):
    # Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def _dct2(img):
    return cv2.dct(img.astype(np.float32))

def _dwt2_levels(img, wavelet='haar'):
    cA1, (cH1, cV1, cD1) = pywt.dwt2(img, wavelet)
    cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, wavelet)
    return cA2, (cH2, cV2, cD2), (cA1, (cH1, cV1, cD1))

def _extract_bits_LL2_simple(LL2_like, LL2_host, alpha, sigma_host):
    """Inversione embedding A: E = (LL2_like - LL2_host) / (alpha * sigma_host) -> bit = (E >= 0)"""
    E = (LL2_like[:32, :32] - LL2_host[:32, :32]) / (alpha * (sigma_host + 1e-12))
    bits = (E >= 0).astype(np.uint8)  # (32,32) in {0,1}
    return bits

def detection(input1, input2, input3):
    # 1) leggi le tre immagini (512x512, grayscale)
    host = cv2.imread(input1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    wm   = cv2.imread(input2, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    att  = cv2.imread(input3, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if host is None or wm is None or att is None or host.shape != (512,512) or wm.shape != (512,512) or att.shape != (512,512):
        raise ValueError("All images must be readable 512x512 grayscale.")

    # 2) DCT
    Hd = _dct2(host); Wd = _dct2(wm); Ad = _dct2(att)

    # 3) DWT a 2 livelli
    H_LL2, _, _ = _dwt2_levels(Hd)
    W_LL2, _, _ = _dwt2_levels(Wd)
    A_LL2, _, _ = _dwt2_levels(Ad)

    # 4) estrazione bit (A: sigma globale su LL2 host)
    sigma_host = float(H_LL2.std())
    bits_w = _extract_bits_LL2_simple(W_LL2, H_LL2, ALPHA, sigma_host)
    bits_a = _extract_bits_LL2_simple(A_LL2, H_LL2, ALPHA, sigma_host)

    # 5) similarità su rappresentazione bipolare {-1,+1}
    X  = (bits_w * 2 - 1).astype(np.float32).ravel()
    Xs = (bits_a * 2 - 1).astype(np.float32).ravel()
    sim = float(similarity(X, Xs))

    # 6) decisione + WPSNR
    output1 = 1 if sim >= TAU else 0
    output2 = float(wpsnr(wm.astype(np.uint8), att.astype(np.uint8)))
    return output1, output2
