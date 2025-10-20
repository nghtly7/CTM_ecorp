import numpy as np
import cv2
import pywt
from wpsnr import wpsnr  # richiesto dal regolamento

# Parametri fissati
WATERMARK_SIZE = 1024
BLOCK_SIZE = 8
THRESHOLD = 0.75  # τ

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def psnr(original, watermarked):
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def _extract_bits_from_coeffs(coeff_array):
    """
    Estrae WATERMARK_SIZE bit scorrendo i coefficienti per blocchi 8x8
    senza saltare nessun coefficiente (stessa scansione dell'embedding).
    Bit = LSB della quantizzazione: round(coeff/Q) & 1, con Q = max(0.1, |coeff|*0.01).
    """
    bits = []
    H, W = coeff_array.shape

    for i in range(0, H, BLOCK_SIZE):
        for j in range(0, W, BLOCK_SIZE):
            if len(bits) >= WATERMARK_SIZE:
                break
            block = coeff_array[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE].flatten()
            for coeff_val in block:
                if len(bits) >= WATERMARK_SIZE:
                    break
                # stesso quantizer dell'embedding
                Q = max(0.1, abs(coeff_val) * 0.01)
                quantized = int(np.round(coeff_val / Q))
                bits.append(quantized & 1)

    # tronca alla lunghezza esatta
    if len(bits) > WATERMARK_SIZE:
        bits = bits[:WATERMARK_SIZE]
    return np.array(bits, dtype=np.uint8)

def _extract_from_image(img_float):
    """
    DWT a 3 livelli, poi lettura LH2 → HL2 (ordine identico all'embedding).
    """
    # Livello 1
    LL1, (LH1, HL1, HH1) = pywt.dwt2(img_float, 'db4', mode='symmetric')
    # Livello 2
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'db4', mode='symmetric')

    # Prima LH2
    bits = _extract_bits_from_coeffs(LH2)
    # Poi HL2 se mancano bit
    if len(bits) < WATERMARK_SIZE:
        remaining = WATERMARK_SIZE - len(bits)
        more = _extract_bits_from_coeffs(HL2)
        if len(more) > remaining:
            more = more[:remaining]
        bits = np.concatenate([bits, more])
    return bits

def detection(input1, input2, input3):
    """
    input1: path immagine originale (non usata per la decisione, ma presente per interfaccia)
    input2: path immagine watermarked
    input3: path immagine attaccata

    return:
      output1: 1 se watermark presente, 0 altrimenti (sim >= THRESHOLD)
      output2: wpsnr(watermarked, attacked)
    """
    #original= cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    att = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)

    # conversione a float per DWT
    #original_f = original.astype(np.float64)
    wm_f = wm.astype(np.float64)
    att_f = att.astype(np.float64)

    # Estrazione watermark
    W_ex = _extract_from_image(wm_f)   # da watermarked
    W_att = _extract_from_image(att_f) # da attacked

    # Similarità (frazione di bit uguali)
    sim = similarity(W_ex, W_att)
    print("similarity att-marked: ",sim)
    
    orig = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    wm   = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)

    print("PSNR origin-marked: ", psnr(orig, wm), "dB")

    # Decisione
    output1 = 1 if sim >= THRESHOLD else 0

    # WPSNR tra watermarked e attacked
    output2 = wpsnr(wm, att)

    return output1, output2
