#!/usr/bin/env python3
"""
detection.py

Detection per l'algoritmo DWT(1)->HL1->DCT 8x8->mid-band + PN (compatibile con embedding.py).
Richiede: opencv-python, numpy, pywt

Firma funzione principale:
    detection(input1, input2, input3, key=12345) -> (det:int, wpsnr:float)

Note:
 - legge threshold_tau.txt per τ (una riga con il valore); fallback tau=0.9
 - allinea l'immagine attaccata alla watermarked usando SIFT/ORB + omografia
 - restituisce det (1/0) e WPSNR(watermarked, attacked_aligned)
"""
import os
import numpy as np
import cv2
import pywt

# ---------------------------
# Utilities (coerenti con embedding.py)
# ---------------------------
def _midband_mask_8x8():
    mask = np.zeros((8,8), dtype=bool)
    coords = [
        (0,3),(0,4),(1,2),(1,3),(1,4),(1,5),
        (2,1),(2,2),(2,3),(2,4),(2,5),(3,0),
        (3,1),(3,2),(3,3),(3,4),(4,0),(4,1),
        (4,2),(4,3),(5,1),(5,2)
    ]
    for (i,j) in coords:
        mask[i,j] = True
    return mask

_MID_MASK = _midband_mask_8x8()

def _pn_sequences(seed, L):
    rng = np.random.RandomState(int(seed))
    pn0 = rng.randn(L).astype(np.float32)
    pn1 = rng.randn(L).astype(np.float32)
    pn0 = np.sign(pn0); pn0[pn0==0]=1
    pn1 = np.sign(pn1); pn1[pn1==0]=1
    return pn0, pn1

def _imread_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere immagine: {path}")
    if img.shape != (512,512):
        raise ValueError("L'immagine deve essere 512x512 grayscale.")
    return img.astype(np.uint8)

def _wpsnr_uint8(img1, img2):
    """
    WPSNR semplificata compatibile con input uint8 in [0,255].
    Restituisce un valore in dB; se identiche restituisce 100.
    """
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 100.0
    return 10.0 * np.log10((255.0 ** 2) / mse)

def _load_tau(path="threshold_tau.txt", default=0.9):
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return float(default)

# ---------------------------
# Estrattore di bit (coerente con embedding.py)
# ---------------------------
def _extract_bits_from_image_uint8(img_uint8, key):
    """
    img_uint8: immagine 512x512 uint8
    key: seed PN
    Ritorna vettore di 1024 bit (dtype uint8)
    """
    img = img_uint8.astype(np.float32)
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    band = HL.astype(np.float32)   # 256x256
    H, W = band.shape
    nby, nbx = H // 8, W // 8
    if nby * nbx != 1024:
        raise RuntimeError("HL1 blocks != 1024; immagine o wavelet non corretti.")
    L = int(_MID_MASK.sum())
    pn0, pn1 = _pn_sequences(key, L)

    bits = np.zeros(nby * nbx, dtype=np.uint8)
    k = 0
    for by in range(0, H, 8):
        for bx in range(0, W, 8):
            blk = band[by:by+8, bx:bx+8]
            d = cv2.dct(blk.astype(np.float32))
            mid = d[_MID_MASK].astype(np.float32)
            c0 = float(np.dot(mid, pn0))
            c1 = float(np.dot(mid, pn1))
            bits[k] = 1 if c1 > c0 else 0
            k += 1
    return bits

def _normalized_correlation(b1, b2):
    x = b1.astype(np.float32)
    y = b2.astype(np.float32)
    if x.size != y.size:
        return 0.0
    num = float(np.dot(x, y))
    den = np.sqrt(float(np.dot(x, x) * np.dot(y, y))) + 1e-12
    return num / den

# ---------------------------
# Allineamento (SIFT/ORB + findHomography)
# ---------------------------
def _register_to_reference_img_uint8(ref_img, atk_img):
    """
    Allinea atk_img a ref_img. Entrambi uint8 grayscale 512x512.
    Ritorna l'immagine attaccata trasformata (uint8). Se non possibile ritorna atk_img originale.
    """
    # try SIFT, fallback ORB
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create()
    else:
        detector = cv2.ORB_create(nfeatures=2000)

    kp1, des1 = detector.detectAndCompute(ref_img, None)
    kp2, des2 = detector.detectAndCompute(atk_img, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return atk_img

    # FLANN for SIFT (float descriptors) / BF for ORB
    try:
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
        if des2.dtype != np.float32:
            des2 = des2.astype(np.float32)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=64)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
    except Exception:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING if des1.dtype==np.uint8 else cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) < 8:
        return atk_img

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
    if H is None:
        return atk_img
    aligned = cv2.warpPerspective(atk_img, H, (ref_img.shape[1], ref_img.shape[0]))
    return aligned

# ---------------------------
# Funzione detection (firma richiesta)
# ---------------------------
def detection(input1, input2, input3, key=12345):
    """
    input1: originale (str) -- non usato per estrazione ma incluso nella firma
    input2: watermarked (str)
    input3: attacked (str)
    key: seed PN

    return: (det:int, wpsnr:float)
    """
    # Caricamento immagini
    Iw = _imread_gray(input2)   # watermarked uint8
    Ia = _imread_gray(input3)   # attacked uint8

    # Allineamento: registriamo Ia sull'immagine watermarked Iw
    Ia_aligned = _register_to_reference_img_uint8(Iw, Ia)

    # Estrai i 1024 bit da watermarked e attacked_aligned
    W_w = _extract_bits_from_image_uint8(Iw, key)
    W_a = _extract_bits_from_image_uint8(Ia_aligned, key)

    # Similarità (Normalized Correlation)
    sim = _normalized_correlation(W_w, W_a)

    # Decisione con tau caricato da file (fallback)
    tau = _load_tau("threshold_tau.txt", default=0.9)
    det = 1 if sim >= tau else 0

    # WPSNR tra watermarked e attacked_aligned
    w = _wpsnr_uint8(Iw, Ia_aligned)

    return int(det), float(w)

