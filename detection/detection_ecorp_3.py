# detection_ecorp_3.py
import numpy as np
import cv2
import pywt
import os

def _compute_wpsnr_fallback(img1, img2):
    # fallback PSNR in dB if wpsnr function not available
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def _try_import_wpsnr():
    try:
        from wpsnr import wpsnr
        return wpsnr
    except Exception:
        return None

def _extract_bits_from_coeff_array(coeff_array, watermark_size, block_size=8):
    """
    Estrae bit dalla matrice dei coefficienti (scansionando blocchi block_size x block_size,
    appiattendo e prendendo la LSB della quantizzazione di ogni coefficiente significativo).
    Restituisce una lista di bit fino a watermark_size.
    """
    bits = []
    H, W = coeff_array.shape
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            if len(bits) >= watermark_size:
                break
            end_i = min(i + block_size, H)
            end_j = min(j + block_size, W)
            block = coeff_array[i:end_i, j:end_j]
            flat = block.flatten()
            for coeff_val in flat:
                if len(bits) >= watermark_size:
                    break
                # ignore very small coefficients (same condition used in embedding)
                if abs(coeff_val) <= 1e-6:
                    # append a placeholder bit (optional): we skip these coefficients
                    continue
                # quantization step used in embedding: Q = max(0.1, abs(coeff)*0.01)
                Q = max(0.1, abs(coeff_val) * 0.01)
                quantized = int(np.round(coeff_val / Q))
                bit = (quantized & 1)
                bits.append(bit)
    # if we didn't extract enough bits, pad with zeros
    if len(bits) < watermark_size:
        bits.extend([0] * (watermark_size - len(bits)))
    return np.array(bits[:watermark_size], dtype=np.uint8)

def detection(original_path: str, watermarked_path: str, attacked_path: str, watermark_path: str = "mark.npy",
              block_size=8, watermark_size=1024, presence_threshold=0.80):
    """
    Detection per l'implementazione embedding_ecorp_3.py.
    :param original_path: percorso immagine originale (non watermarkata)
    :param watermarked_path: percorso immagine watermarked (opzionale ma utile per confronto)
    :param attacked_path: percorso immagine attaccata (da cui estrarre)
    :param watermark_path: percorso file .npy contenente watermark originale (0/1, length = watermark_size)
    :param presence_threshold: soglia di accuratezza (fra 0 e 1) per decidere presenza watermark
    :return: (dec, wpsnr_value, bit_accuracy) dove dec è 1 (presente) o 0 (assente)
    """
    # carica immagini in grayscale
    orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    wm_img = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
    att = cv2.imread(attacked_path, cv2.IMREAD_GRAYSCALE)

    if orig is None:
        raise ValueError(f"Cannot load original image from {original_path}")
    if wm_img is None:
        raise ValueError(f"Cannot load watermarked image from {watermarked_path}")
    if att is None:
        raise ValueError(f"Cannot load attacked image from {attacked_path}")

    # converti in float64 per DWT
    orig_f = orig.astype(np.float64)
    wm_f = wm_img.astype(np.float64)
    att_f = att.astype(np.float64)

    # funzione per ottenere LH2 e HL2 (stesso ordine usato nell'embedding: livello3 -> LH3/HL3/HH3, poi IDWT parziale ecc)
    def get_level2_detail_coeffs(img_float):
        # Level1
        coeffs1 = pywt.dwt2(img_float, 'db4', mode='symmetric')
        LL1, (LH1, HL1, HH1) = coeffs1
        # Level2
        coeffs2 = pywt.dwt2(LL1, 'db4', mode='symmetric')
        LL2, (LH2, HL2, HH2) = coeffs2
        # Level3
        coeffs3 = pywt.dwt2(LL2, 'db4', mode='symmetric')
        LL3, (LH3, HL3, HH3) = coeffs3
        # return the level2 detail coefficients LH2, HL2 and the level3 subbands as used in embedding
        return LH2, HL2, (LL1, (LH1, HL1, HH1)), (LL2, (LH2, HL2, HH2)), (LL3, (LH3, HL3, HH3))

    # estrai coefficienti da watermarked e attacked
    LH2_wm, HL2_wm, *_ = get_level2_detail_coeffs(wm_f)
    LH2_att, HL2_att, *_ = get_level2_detail_coeffs(att_f)

    # estrazione: stesso ordine usato in embedding -> LH2 poi HL2
    extracted_bits_from_wm = []
    extracted_bits_from_att = []

    extracted_bits_from_wm.extend(_extract_bits_from_coeff_array(LH2_wm, watermark_size, block_size))
    if len(extracted_bits_from_wm) < watermark_size:
        remaining = watermark_size - len(extracted_bits_from_wm)
        extracted_bits_from_wm.extend(_extract_bits_from_coeff_array(HL2_wm, remaining, block_size))

    extracted_bits_from_att.extend(_extract_bits_from_coeff_array(LH2_att, watermark_size, block_size))
    if len(extracted_bits_from_att) < watermark_size:
        remaining = watermark_size - len(extracted_bits_from_att)
        extracted_bits_from_att.extend(_extract_bits_from_coeff_array(HL2_att, remaining, block_size))

    # make numpy arrays trimmed to watermark_size
    extracted_bits_from_wm = np.array(extracted_bits_from_wm[:watermark_size], dtype=np.uint8)
    extracted_bits_from_att = np.array(extracted_bits_from_att[:watermark_size], dtype=np.uint8)

    # salva per debug
    try:
        np.save("extracted_mark_wm.npy", extracted_bits_from_wm)
        np.save("extracted_mark_att.npy", extracted_bits_from_att)
    except Exception:
        pass

    # carica watermark originale se disponibile
    original_mark = None
    if os.path.exists(watermark_path):
        try:
            original_mark = np.load(watermark_path).astype(np.uint8)
            if original_mark.size != watermark_size:
                # se il file ha formato diverso, proviamo a ridimensionare/trim
                original_mark = original_mark.flatten()[:watermark_size].astype(np.uint8)
        except Exception:
            original_mark = None

    # calcola accuratezza bit se abbiamo la mark originale
    bit_accuracy = None
    dec = 0
    if original_mark is not None:
        matches = (extracted_bits_from_att == original_mark)
        bit_accuracy = float(np.sum(matches)) / float(watermark_size)
        dec = 1 if bit_accuracy >= presence_threshold else 0
    else:
        # se non abbiamo la mark originale, facciamo confronto tra estrazione da wm e attaccata
        matches = (extracted_bits_from_att == extracted_bits_from_wm)
        bit_accuracy = float(np.sum(matches)) / float(watermark_size)
        # una soglia più conservativa se paragoniamo estrazioni: 0.75
        dec = 1 if bit_accuracy >= 0.75 else 0

    # ritorna decisione, wpsnr e accuratezza bit (utile per debug/report)
    return dec, bit_accuracy
