from wpsnr import wpsnr
import numpy as np
import cv2
import pywt

TAU = 0.80
ALPHA = 0.2  # deve combaciare con l'embedding (stesso alpha usato in B)

def similarity(X, X_star):
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def _dct2(img):
    return cv2.dct(img.astype(np.float32))

def _dwt2_levels(img, wavelet='haar'):
    cA1, (cH1, cV1, cD1) = pywt.dwt2(img, wavelet)
    cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, wavelet)
    return cA2, (cH2, cV2, cD2), (cA1, (cH1, cV1, cD1))

def _extract_bits_LL2_blocknorm(LL2_like, LL2_host, alpha):
    """Inversione embedding B: per ogni blocco 8x8 uso sigma del blocco host."""
    bits = np.zeros((32,32), dtype=np.uint8)
    for by in range(4):
        for bx in range(4):
            y0, x0 = by*8, bx*8
            patch_like = LL2_like[y0:y0+8, x0:x0+8]
            patch_host = LL2_host[y0:y0+8, x0:x0+8]
            sigma_blk  = float(patch_host.std()) + 1e-12
            E_blk = (patch_like - patch_host) / (alpha * sigma_blk)
            bits[y0:y0+8, x0:x0+8] = (E_blk >= 0).astype(np.uint8)
    return bits

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

    bits_w = _extract_bits_LL2_blocknorm(W_LL2, H_LL2, ALPHA)
    bits_a = _extract_bits_LL2_blocknorm(A_LL2, H_LL2, ALPHA)

    X  = (bits_w * 2 - 1).astype(np.float32).ravel()
    Xs = (bits_a * 2 - 1).astype(np.float32).ravel()
    sim = float(similarity(X, Xs))

    output1 = 1 if sim >= TAU else 0
    output2 = float(wpsnr(wm.astype(np.uint8), att.astype(np.uint8)))
    return output1, output2
