import os
import numpy as np
import cv2
import pywt


#* tutti sopra 58

def _load_watermark_bits(path_or_array):
    """
    Carica il watermark come vettore binario di lunghezza 1024.
    Accetta:
      - numpy array già passato
      - file .npy con array 1D/2D di 0/1
      - file di testo con 0/1 (separati da spazi o senza spazi/righe)
    """
    if isinstance(path_or_array, np.ndarray):
        bits = path_or_array.astype(np.uint8).flatten()
    else:
        p = str(path_or_array)
        ext = os.path.splitext(p)[1].lower()
        if ext == ".npy":
            bits = np.load(p).astype(np.uint8).flatten()
        else:
            # prova a leggere come testo (accetta "0101..." o "0 1 0 1 ..." o righe)
            with open(p, "r") as f:
                s = f.read().strip().replace("\n", "").replace("\r", "").replace(" ", "")
            if set(s) - set("01"):
                raise ValueError("File watermark non valido: deve contenere solo 0/1.")
            bits = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord('0')
    if bits.size != 1024:
        raise ValueError(f"Il watermark deve avere 1024 bit, trovato {bits.size}.")
    return bits.astype(np.uint8)

def _block_view(arr, block_shape=(8,8)):
    """
    Ritorna una vista 4D dell'immagine come blocchi (nB_r, nB_c, bh, bw).
    Richiede che le dimensioni siano multipli della dimensione blocco.
    """
    h, w = arr.shape
    bh, bw = block_shape
    assert h % bh == 0 and w % bw == 0
    arr_reshaped = arr.reshape(h//bh, bh, w//bw, bw)
    return np.swapaxes(arr_reshaped, 1, 2)  # (nBr, nBc, bh, bw)

def _inverse_block_view(blocks):
    """
    Inverte _block_view: da (nBr, nBc, bh, bw) a (H, W)
    """
    nBr, nBc, bh, bw = blocks.shape
    arr = np.swapaxes(blocks, 1, 2).reshape(nBr*bh, nBc*bw)
    return arr

def _adaptive_delta(dct_block, base_delta=1.5, gamma=0.5):
    """
    Calcola un delta adattivo in funzione dell'energia del blocco in DCT (escluso DC).
    base_delta controlla l'entità media della modifica (in unità di coeff. DCT).
    gamma controlla quanto forte cresce con l'energia (0 = costante).
    """
    # energia media escludendo DC
    energy = np.mean(np.abs(dct_block)) - np.abs(dct_block[0,0]) / (dct_block.size)
    energy = max(energy, 1e-6)
    # normalizzazione grossolana: usa una funzione concava per non esplodere
    return float(base_delta * (energy ** gamma))

def _embed_bit_in_dct_block(dct_block, bit, pos1=(2,3), pos2=(3,2), base_delta=1.5, gamma=0.5):
    """
    Modifica i due coefficienti mid-band (pos1, pos2) per codificare il bit con margine delta adattivo.
    Schema 'comparativo' (tipo c1 - c2 >= delta per bit=1, invertito per bit=0).
    """
    c1 = dct_block[pos1]
    c2 = dct_block[pos2]
    delta = _adaptive_delta(dct_block, base_delta=base_delta, gamma=gamma)

    if bit == 1:
        # Imporre c1 - c2 >= delta
        if (c1 - c2) < delta:
            # correzione minima a somma zero per non alterare troppo l'energia
            corr = (delta - (c1 - c2)) / 2.0
            c1 += corr
            c2 -= corr
    else:
        # Imporre c2 - c1 >= delta
        if (c2 - c1) < delta:
            corr = (delta - (c2 - c1)) / 2.0
            c2 += corr
            c1 -= corr

    dct_block[pos1] = c1
    dct_block[pos2] = c2

def embedding(input1, input2="ecorp.npy",
              wavelet="haar",
              subband="HL",
              block_size=8,
              base_delta=1.5,
              gamma=0.5):
    """
    Embedding DWT->DCT (blocchi 8x8) con modifica selettiva di coefficienti mid-band (Ernawan et al., 2021).
    - input1: path dell'immagine BMP 512x512 grayscale
    - input2: path (o array) del watermark 1024-bit (npy/testo) con {0,1}
    Parametri:
      wavelet: 'haar' (consigliato)
      subband: 'HL' o 'LH' (HL spesso funziona bene)
      block_size: 8 (deve dividere 256)
      base_delta: intensità media della modifica in DCT (più alto = più robusto, minore qualità)
      gamma: quanto rendere adattiva la modifica rispetto all'energia del blocco (0..1)
    Ritorna:
      output1: immagine watermarked come array uint8 (512x512)
    NOTE:
      - Nessuna stampa o GUI (rispetta le regole).
      - La qualità (WPSNR) dipende da base_delta e gamma: tararla con il vostro set.
    """
    # --- carica immagine ---
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {input1}")
    if img.shape != (512, 512):
        raise ValueError("L'immagine deve essere 512x512 grayscale.")
    img_f = img.astype(np.float32)

    # --- carica watermark 1024 bit ---
    wm_bits = _load_watermark_bits(input2)  # (1024,)

    # --- DWT 1 livello ---
    coeffs2 = pywt.dwt2(img_f, wavelet)
    LL, (LH, HL, HH) = coeffs2

    # scegli la subband di lavoro (256x256)
    if subband.upper() == "HL":
        sb = HL.copy()
    elif subband.upper() == "LH":
        sb = LH.copy()
    else:
        raise ValueError("subband deve essere 'HL' o 'LH'.")

    if sb.shape != (256, 256):
        raise ValueError("Sottobanda inattesa (deve essere 256x256).")

    # --- blocchi 8x8 sulla subband ---
    if (256 % block_size) != 0:
        raise ValueError("block_size deve dividere 256.")
    blocks = _block_view(sb, (block_size, block_size))  # (32,32,8,8)
    nBr, nBc, bh, bw = blocks.shape
    if (nBr * nBc) != 1024:
        raise RuntimeError("La sottobanda non produce esattamente 1024 blocchi.")

    # --- DCT, embedding per blocco, IDCT ---
    idx = 0
    for r in range(nBr):
        for c in range(nBc):
            block = blocks[r, c, :, :].astype(np.float32)
            # DCT 2D
            dctB = cv2.dct(block)
            # inserisci bit su due coeff mid-band
            bit = int(wm_bits[idx])
            _embed_bit_in_dct_block(dctB, bit, pos1=(2,3), pos2=(3,2),
                                    base_delta=base_delta, gamma=gamma)
            # IDCT
            idctB = cv2.idct(dctB)
            blocks[r, c, :, :] = idctB
            idx += 1

    # ricompone la subband
    sb_wm = _inverse_block_view(blocks)

    # sostituisci nella terna
    if subband.upper() == "HL":
        HL_wm = sb_wm
        out_coeffs = (LL, (LH, HL_wm, HH))
    else:
        LH_wm = sb_wm
        out_coeffs = (LL, (LH_wm, HL, HH))

    # --- IDWT ---
    img_wm = pywt.idwt2(out_coeffs, wavelet)

    # --- clamp & tipo ---
    img_wm = np.clip(img_wm, 0, 255).astype(np.uint8)
    return img_wm
