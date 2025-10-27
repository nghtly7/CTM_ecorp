#!/usr/bin/env python3
"""
tuning_per_image_v2.py — WPSNR-first, stop alla prima combo ≥ 58 dB.
- Ordina le combinazioni dalla più forte alla più debole
- Prova anche scale di alpha (1.0 → 0.3) per indebolire progressivamente
- Nessuna detection: solo WPSNR
- Log compatto (L2): stampa la combo trovata
- Salva in watermarked_images/<nome>.*
"""

import os, sys, itertools, importlib
import numpy as np
import cv2

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(ROOT_DIR)

IMAGES_DIR = os.path.join(ROOT_DIR, "sample-images")
WM_DIR     = os.path.join(ROOT_DIR, "watermarked_images")
os.makedirs(WM_DIR, exist_ok=True)

WATERMARK_PATH = os.path.join(ROOT_DIR, "ecorp.npy")

# ---- Import moduli ----
from wpsnr import wpsnr as WPSNR
emb_mod   = importlib.import_module("paper_embedding_v1_2")
embedding = getattr(emb_mod, "embedding")

# ---- Parametri & Ordini (forte → debole) ----
BETAS   = [0.8, 0.6, 0.4]        # più alto = più forte
GAMMAS  = [0.6, 0.4, 0.2]        # più alto = più forte (penalizza poco attack_map)
QS      = [0.6, 0.8, 1.0]        # più basso = più forte (snap più forte)
SOFT_TS = [0.10, 0.15, 0.20]     # più basso = più forte (meno blocchi “piatti”)
SOFT_KS = [0.6, 0.4, 2]          # più alto = più “skip/attenuazione” → qui lo mettiamo tra i primi per coerenza con tuo setup

ALPHA_SCALES = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]  # 1.0 = più forte, 0.3 = più debole
WPSNR_MIN = 58.0

def list_images(folder):
    exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif")
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]

def main():
    imgs = list_images(IMAGES_DIR)
    if not imgs:
        print("Nessuna immagine trovata in sample-images/")
        return

    for img_name in imgs:
        img_path = os.path.join(IMAGES_DIR, img_name)
        I_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if I_orig is None:
            print(f'[!] Impossibile leggere "{img_name}"')
            continue

        print(f'\n[TUNING] Testing image "{img_name}"')

        found = False

        # griglia ordinata forte→debole
        grid = list(itertools.product(BETAS, GAMMAS, QS, SOFT_TS, SOFT_KS))

        for a_scale in ALPHA_SCALES:
            # cicla le combinazioni forti→deboli
            for (beta, gamma, Q, soft_t, soft_k) in grid:
                # imposta i global dell'embedding
                setattr(emb_mod, "beta",   beta)
                setattr(emb_mod, "gamma",  gamma)
                setattr(emb_mod, "Q",      Q)
                setattr(emb_mod, "soft_t", soft_t)
                setattr(emb_mod, "soft_k", soft_k)

                try:
                    # embedding con alpha scale
                    Iw = embedding(img_path, WATERMARK_PATH, alpha_scale=a_scale)
                except Exception:
                    continue

                try:
                    wps_clean = float(WPSNR(I_orig, Iw))
                except Exception:
                    continue

                if wps_clean >= WPSNR_MIN:
                    # log L2: compatto ma informativo
                    print(f'    FOUND: {wps_clean:.2f} dB  (alpha={a_scale:.2f}, beta={beta}, gamma={gamma}, Q={Q}, soft_t={soft_t}, soft_k={soft_k})')
                    out_path = os.path.join(WM_DIR, img_name)
                    cv2.imwrite(out_path, Iw)
                    print(f'[SAVED] {img_name}')
                    found = True
                    break

            if found:
                break

        if not found:
            print(f'wpsnr minimo non raggiunto per l\'immagine "{img_name}"')

if __name__ == "__main__":
    main()
