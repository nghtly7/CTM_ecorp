import numpy as np
import cv2
import pywt

def _block_view(arr, block_shape=(8,8)):
    h, w = arr.shape
    bh, bw = block_shape
    assert h % bh == 0 and w % bw == 0
    arr_reshaped = arr.reshape(h//bh, bh, w//bw, bw)
    return np.swapaxes(arr_reshaped, 1, 2)

def _extract_bits_from_image(img_array, wavelet="haar", subband="HL", block_size=8):
    coeffs2 = pywt.dwt2(img_array.astype(np.float32), wavelet)
    LL, (LH, HL, HH) = coeffs2

    sb = HL if subband.upper() == "HL" else LH
    blocks = _block_view(sb, (block_size, block_size))  # (32,32,8,8)

    wm_bits = []
    for r in range(blocks.shape[0]):
        for c in range(blocks.shape[1]):
            block = blocks[r, c, :, :].astype(np.float32)
            dctB = cv2.dct(block)
            c1 = dctB[2,3]
            c2 = dctB[3,2]
            wm_bits.append(1 if c1 > c2 else 0)
    return np.array(wm_bits, dtype=np.uint8)  # shape (1024,)

def _compute_similarity(w1, w2, method="ber"):
    if method == "ber":
        return 1.0 - np.count_nonzero(w1 != w2) / len(w1)
    elif method == "dot":
        return np.dot(w1, w2) / max(1, np.linalg.norm(w1) * np.linalg.norm(w2))
    else:
        raise ValueError("Unknown similarity method.")

def _wpsnr(orig, distorted):
    orig = orig.astype(np.float32)
    distorted = distorted.astype(np.float32)
    mse = np.mean((orig - distorted) ** 2)
    if mse == 0:
        return 99.0
    PIXEL_MAX = 255.0
    return 10 * np.log10((PIXEL_MAX**2) / mse)

# === detection function ===

def detection(input1, input2, input3):
    """
    Detection non-blind:
      - input1: path immagine originale (NON usata per detection, ma richiesta dalla funzione)
      - input2: path immagine watermarked
      - input3: path immagine attaccata
    Return:
      output1: 1 se watermark rilevato, 0 se rimosso
      output2: WPSNR tra input2 e input3
    """
    # Parametri coerenti con embedding
    wavelet = "haar"
    subband = "HL"
    block_size = 8
    threshold_tau = 0.80  # <-- va ottimizzato con ROC

    # --- leggi immagini ---
    wm_img = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    att_img = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)
    if wm_img is None or att_img is None:
        raise FileNotFoundError("Immagine mancante in input2 o input3.")
    if wm_img.shape != (512,512) or att_img.shape != (512,512):
        raise ValueError("Le immagini devono essere 512x512 grayscale.")

    # --- estrazione ---
    w1 = _extract_bits_from_image(wm_img, wavelet, subband, block_size)
    w2 = _extract_bits_from_image(att_img, wavelet, subband, block_size)

    # --- confronto ---
    similarity = _compute_similarity(w1, w2, method="ber")

    # --- decisione ---
    output1 = int(similarity >= threshold_tau)
    output2 = float(_wpsnr(wm_img, att_img))

    return output1, output2
