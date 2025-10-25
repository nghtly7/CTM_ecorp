#!/usr/bin/env python3
"""
embed_watermark.py

Embedding DWT(1)->HL1->DCT 8x8->mid-band + PN sequences
- Input immagine: BMP 512x512 grayscale
- Watermark: .npy con esattamente 1024 bit (dtype any, verrà convertito a uint8)
- Restituisce immagine watermarked (numpy.float32) e, se dato, la salva su disco.

Funzione principale:
    embedding(input_image_path, watermark_npy_path, key=12345, alpha=0.4, save_path=None)

Dipendenze:
    opencv-python, numpy, pywt

Esempio:
    Iw = embedding("orig.bmp", "wm.npy", key=2025, alpha=0.35, save_path="watermarked.bmp")
"""
import os
import numpy as np
import cv2
import pywt

# ---------------------------
# Utils minimi
# ---------------------------
def _imread_gray_bmp(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere l'immagine: {path}")
    if img.ndim != 2:
        raise ValueError("L'immagine deve essere in scala di grigi.")
    if img.shape != (512, 512):
        raise ValueError("L'immagine deve essere 512x512.")
    return img.astype(np.float32)

def _imwrite_gray_bmp(path, img_float):
    img = np.clip(np.rint(img_float), 0, 255).astype(np.uint8)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Errore salvataggio immagine: {path}")

def _dct2(block):
    return cv2.dct(block.astype(np.float32))

def _idct2(block):
    return cv2.idct(block.astype(np.float32))

# ---------------------------
# Mid-band mask 8x8 (tipica)
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

# ---------------------------
# PN sequences generator
# ---------------------------
def _pn_sequences(seed, L):
    rng = np.random.RandomState(int(seed))
    pn0 = rng.randn(L).astype(np.float32)
    pn1 = rng.randn(L).astype(np.float32)
    # semplifichiamo a ±1 per spread-spectrum robusto
    pn0 = np.sign(pn0); pn0[pn0==0]=1
    pn1 = np.sign(pn1); pn1[pn1==0]=1
    return pn0, pn1

# ---------------------------
# Processo di embedding (band HL1)
# ---------------------------
def _block_process_HL1(img, func):
    """
    Applica func(dct_block, (by,bx)) su ogni blocco 8x8 della banda HL1.
    Ritorna immagine ricostruita (IDWT).
    """
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    band = HL.copy().astype(np.float32)    # HL1: dimensione 256x256

    H, W = band.shape
    out_band = band.copy()

    for by in range(0, H, 8):
        for bx in range(0, W, 8):
            blk = band[by:by+8, bx:bx+8]
            if blk.shape != (8,8):
                continue
            d = _dct2(blk)
            d_new = func(d, (by, bx))
            out_band[by:by+8, bx:bx+8] = _idct2(d_new)

    coeffs2_w = (LL, (LH, out_band, HH))
    rec = pywt.idwt2(coeffs2_w, 'haar')
    return np.clip(rec, 0.0, 255.0)

# ---------------------------
# Funzione principale: embedding
# ---------------------------
def embedding(input_image_path, watermark_npy_path="ecorp.npy", key=12345, alpha=0.4):
    """
    Esegue l'embedding del watermark.

    Parametri:
        input_image_path : str  -> path immagine originale (BMP 512x512 grayscale)
        watermark_npy_path: str  -> path file .npy contenente 1024 bit
        key               : int  -> seed per generare le PN sequences
        alpha             : float-> coefficiente di forza di embedding (λ)
        save_path         : str/None -> se fornito salva l'immagine watermarked qui

    Ritorno:
        numpy.ndarray float32 immagine watermarked (512x512)
    """
    # Caricamento e controlli
    I = _imread_gray_bmp(input_image_path)

    if not os.path.exists(watermark_npy_path):
        raise FileNotFoundError(f"Watermark file non trovato: {watermark_npy_path}")
    Wbits = np.load(watermark_npy_path)
    Wbits = np.array(Wbits).astype(np.uint8).flatten()
    if Wbits.size != 1024:
        raise ValueError("Watermark deve contenere esattamente 1024 bit.")

    # Preparazione PN e parametri
    L = int(_MID_MASK.sum())
    pn0, pn1 = _pn_sequences(key, L)

    # Funzione che modifica il blocco DCT in base al bit corrispondente
    def _embed_block(dct_blk, idx):
        by, bx = idx
        mid = dct_blk[_MID_MASK].astype(np.float32)
        # scaling locale: media assoluta dei coeff. mid-band (stima adattiva come nel paper)
        m = float(np.mean(np.abs(mid)) + 1e-9)
        # mappatura bit index: blocchi in HL1 sono 32x32 -> ordine row-major
        block_row = by // 8
        block_col = bx // 8
        bit_index = block_row * (256 // 8) + block_col  # 0..1023
        b = int(Wbits[bit_index])
        pn = pn1 if b == 1 else pn0
        dct_blk2 = dct_blk.copy()
        # embedding additivo sui coefficienti mid-band
        dct_blk2[_MID_MASK] = mid + (alpha * m) * pn
        return dct_blk2

    # Applica embedding su HL1 e ricompone immagine
    Iw = _block_process_HL1(I, _embed_block)

    return np.clip(Iw, 0, 255).astype(np.uint8)
