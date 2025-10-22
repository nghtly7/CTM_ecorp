import cv2
import numpy as np
import pywt
from pathlib import Path

# importa funzioni utili dagli attacks (assicurati che attacks.py sia importabile)
from attacks import awgn, blur, median, resizing  # vedi attacks.py per implementazioni. :contentReference[oaicite:5]{index=5}

alpha = 3.0
FIXED_SEED = 42
BLOCK = 4
MASK = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]

def compute_attack_map(img):
    """Costruisce la attack-map sommando |attacked - original| per una lista di attacchi."""
    orig = img.astype(np.float32)
    A = np.zeros_like(orig, dtype=np.float32)

    # lista esempio (puoi regolarla): piccolo AWGN, blur, median, resizing
    # usa parametri leggeri/moderati per non impiegare troppo tempo
    for std in [0.5, 2.0, 5.0]:
        attacked = awgn(orig, std, seed=0)
        A += np.abs(attacked - orig)

    for sigma in [0.5, 1.5]:
        attacked = blur(orig, sigma)
        A += np.abs(attacked - orig)

    for k in [3,5]:
        attacked = median(orig, [k, k])
        A += np.abs(attacked - orig)

    for scale in [0.9, 0.75]:
        attacked = cv2.resize(cv2.resize(orig, (0,0), fx=scale, fy=scale), (orig.shape[1], orig.shape[0]))
        A += np.abs(attacked - orig)

    # normalizza fra 0 e 1 per comodità
    A = (A - A.min()) / (np.ptp(A) + 1e-12)
    
    # smoothing della attack map per stabilizzare l'alpha_block
    A = cv2.GaussianBlur(A, (5,5), 0.8)
    
    return A

def embedding(input1, input2=None):
    I = cv2.imread(input1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if input2 is None:
        rng = np.random.default_rng(FIXED_SEED)
        Wbits = rng.integers(0,2,size=1024,dtype=np.uint8)
        np.save("generated_watermark.npy", Wbits)
    else:
        Wbits = np.load(input2).astype(np.uint8)

    # calcola attack-map (puoi salvare per debug)
    attack_map = compute_attack_map(I)

    # DWT 3 livelli (come prima)
    coeffs = pywt.wavedec2(I, wavelet='db2', level=3)
    (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs[1], coeffs[2], coeffs[3]
    bands = [HL3, LH3, HL2, LH2]

    rng = np.random.default_rng(FIXED_SEED)
    PN0 = rng.standard_normal(len(MASK))
    PN1 = rng.standard_normal(len(MASK))

    # calcola statistiche globali per normalizzazione
    # useremo l'energia media dei blocchi per calibrare alpha_block
    # precompute block energies across tutte le bande (mid-band values)
    # calcola statistiche globali per normalizzazione

    energies = []
    for b in range(4):
        B = bands[b]
        for by in range(0,64,4):
            for bx in range(0,64,4):
                block = B[by:by+4, bx:bx+4]
                vals = np.array([block[u,v] for (u,v) in MASK], dtype=np.float32)
                energies.append(np.linalg.norm(vals))
    energies = np.array(energies)
    E_mean = energies.mean()
    E_ptp = np.ptp(energies) + 1e-12   # ✅ corregge errore NumPy 2.0

    # embedding con alpha adattivo
    idx = 0
    Q = 0.6   # provare valori tra 0.4 e 1.0
    for b in range(4):
        B = bands[b]
        for by in range(0,64,4):
            for bx in range(0,64,4):
                block = B[by:by+4, bx:bx+4].copy()
                alpha_block = alpha # valore default di alpha
                C = cv2.dct(block)
                bit = int(Wbits[idx])
                PN = PN1 if bit==1 else PN0

                vals = np.array([C[u,v] for (u,v) in MASK], dtype=np.float32)
                E = np.linalg.norm(vals)

                img_y = int((by/64.0) * I.shape[0])
                img_x = int((bx/64.0) * I.shape[1])
                h = max(1, I.shape[0]//64)
                w = max(1, I.shape[1]//64)
                y0 = max(0, img_y - h//2); y1 = min(I.shape[0], y0 + h)
                x0 = max(0, img_x - w//2); x1 = min(I.shape[1], x0 + w)
                attack_score = attack_map[y0:y1, x0:x1].mean() if (y1>y0 and x1>x0) else 0.0

                E_norm = (E - E_mean) / (E_ptp)   # può essere negativo; va bene
                if E < 0.25 * E_mean:   # soglia morbida (provabile tra 0.2 e 0.35)
                    alpha_block *= 0.3  # oppure 0.0 per saltarli del tutto

                beta = 0.6
                gamma = 0.8
                alpha_block = alpha * (1.0 + beta * E_norm) * (1.0 - gamma * attack_score)
                alpha_block = float(np.clip(alpha_block, 0.1*alpha, 2.0*alpha))
                
                #* da capire meglio questo blocco 
                # soft–scaling per blocchi piatti (perceptual skip / attenuation)
                if E < 0.10 * E_mean:          # soglia (consigliato 0.25 – 0.35)   # attualmente soft-skip, per saltare i blocchi bianchi
                     continue #     alpha_block *= 0.25        # attenua (oppure 0.0 se vuoi skip totale)
                #wpsnr diminuisce leggermente, ma robustezza migliora, dipende dall'immagine


                # for k,(u,v) in enumerate(MASK):   # vecchio embedding senza HIR
                #     C[u,v] += alpha_block * PN[k] 
                
                
                # vettore host (mid-band)
                x = np.array([C[u,v] for (u,v) in MASK], dtype=np.float32)
                p = PN1 if bit == 1 else PN0

                # Host-Interference Rejection (proiezione ortogonale)
                den = (x @ x) + 1e-12
                p_ortho = p - ((p @ x) / den) * x
                if np.linalg.norm(p_ortho) < 1e-6:
                    p_use = p
                else:
                    p_use = p_ortho / (np.linalg.norm(p_ortho) + 1e-12)

                # Iniezione watermark ortogonale all’host
                for k, (u,v) in enumerate(MASK):
                    C[u,v] += alpha_block * p_use[k]
                    
                for (u,v) in MASK:
                    C[u,v] = Q * np.round(C[u,v] / Q)

                B[by:by+4, bx:bx+4] = cv2.idct(C)
                idx += 1
        bands[b] = B


    # ricostruzione DWT
    new_coeffs = list(coeffs)
    LH3n, HL3n = bands[1], bands[0]
    LH2n, HL2n = bands[3], bands[2]
    new_coeffs[1] = (LH3n, HL3n, HH3)
    new_coeffs[2] = (LH2n, HL2n, HH2)
    new_coeffs[3] = (LH1, HL1, HH1)
    Iw = pywt.waverec2(new_coeffs, wavelet='db2')
    Iw = np.clip(Iw, 0, 255).astype(np.uint8)

    # opzionale: salva attack_map per debug
    np.save("attack_map.npy", attack_map)
    return Iw
