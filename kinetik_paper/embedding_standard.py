import numpy as np
import cv2
import pywt

def _load_watermark_bits(path: str) -> np.ndarray:
    if path.lower().endswith('.npy'):
        bits = np.load(path).astype(np.uint8).reshape(-1)
    else:
        with open(path, 'r') as f:
            s = f.read()
        s = ''.join(ch for ch in s if ch in '01')
        bits = np.frombuffer(s.encode('ascii'), dtype='S1').astype(np.uint8) - ord('0')
    if bits.size != 1024:
        raise ValueError(f"Watermark must be 1024 bits, got {bits.size}")
    return (bits > 0).astype(np.uint8)

def _dct2(img):
    return cv2.dct(img.astype(np.float32))

def _idct2(coeffs):
    return cv2.idct(coeffs.astype(np.float32))

def _dwt2_levels(img, wavelet='haar'):
    cA1, (cH1, cV1, cD1) = pywt.dwt2(img, wavelet)
    cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, wavelet)
    return cA2, (cH2, cV2, cD2), (cA1, (cH1, cV1, cD1))

def _idwt2_levels(cA2, cH2, cV2, cD2, cA1, cH1, cV1, cD1, wavelet='haar'):
    up1 = pywt.idwt2((cA2, (cH2, cV2, cD2)), wavelet)
    rec = pywt.idwt2((up1, (cH1, cV1, cD1)), wavelet)
    return rec

def embedding(input1, input2="ecorp.npy"):
    """
    Embedding A: direct 1:1 embedding as in the paper.
    input1: path to 512x512 grayscale image
    input2: path to watermark (1024 bits .npy or text)
    returns: watermarked image as np.ndarray uint8 (512x512)
    """
    alpha = 0.2

    host = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if host is None or host.shape != (512, 512):
        raise ValueError("Host must be a 512x512 grayscale image")
    host_f = host.astype(np.float32)

    bits = _load_watermark_bits(input2).reshape(32, 32)

    Hdct = _dct2(host_f)
    LL2, (LH2, HL2, HH2), (LL1, (LH1, HL1, HH1)) = _dwt2_levels(Hdct, wavelet='haar')

    sigma = float(LL2.std()) + 1e-12
    bipolar = (bits * 2 - 1).astype(np.float32)
    pattern = bipolar * sigma

    LL2_w = LL2.copy()
    LL2_w[:32, :32] = LL2_w[:32, :32] + alpha * pattern

    Hdct_w = _idwt2_levels(LL2_w, LH2, HL2, HH2, LL1, LH1, HL1, HH1, wavelet='haar')
    watermarked = _idct2(Hdct_w)

    out = np.clip(np.round(watermarked), 0, 255).astype(np.uint8)
    return out
