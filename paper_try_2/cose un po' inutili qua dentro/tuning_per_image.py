#!/usr/bin/env python3
"""
tuning_per_image.py — versione modificata per WPSNR≥58 first,
poi robustezza massima (criterio A).
"""

import os, sys, csv, json, itertools
import importlib
import numpy as np
import cv2
from time import time

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(ROOT_DIR)

IMAGES_DIR = os.path.join(ROOT_DIR, "sample-images")
WM_DIR     = os.path.join(ROOT_DIR, "watermarked_images")
OUT_DIR    = os.path.join(ROOT_DIR, "tuning_output", "per_image")

os.makedirs(WM_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

WATERMARK_PATH = os.path.join(ROOT_DIR, "ecorp.npy")

# ---- Param grid ----
BETAS   = [0.4, 0.6, 0.8]
GAMMAS  = [0.2, 0.4, 0.6]
QS      = [0.6, 0.8, 1.0]
SOFT_TS = [0.10, 0.15, 0.20]
SOFT_KS = [0.4, 0.6, 2]

# ---- Threshold & policy ----
TAU = 0.75
WPSNR_MIN = 58.0  # soglia fissa richiesta

# ---- Import moduli embedding & detection ----
from wpsnr import wpsnr as WPSNR
emb_mod   = importlib.import_module("paper_embedding_v1_2")
embedding = getattr(emb_mod, "embedding")
det_mod   = importlib.import_module("paper_detection_v1_2")
detection = getattr(det_mod, "detection")

# ---- Attack set (stesso del file originale) ----
rng = np.random.default_rng(0)
def attack_awgn(img, sigma):
    imgf = img.astype(np.float32)
    noise = rng.normal(0, sigma, imgf.shape).astype(np.float32)
    return np.clip(imgf + noise, 0, 255).astype(np.uint8)

def attack_blur(img, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

def attack_jpeg(img, quality):
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

ATTACKS = [
    ("AWGN", lambda im: attack_awgn(im, 3.0)),
    ("BLUR", lambda im: attack_blur(im, 1.0)),
    ("JPEG", lambda im: attack_jpeg(im, 80)),
]

def list_images(folder):
    exts = (".bmp", ".png", ".jpg", ".jpeg")
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]

# ---- MAIN LOOP ----
def main():
    imgs = list_images(IMAGES_DIR)
    if not imgs:
        print("Nessuna immagine trovata in sample-images/")
        return

    for img_name in imgs:
        img_path = os.path.join(IMAGES_DIR, img_name)
        I_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        print(f'\n[TUNING] Testing image "{img_name}"')

        results = []  # (wps_clean, robustness, params, Iw)

        grid = list(itertools.product(BETAS, GAMMAS, QS, SOFT_TS, SOFT_KS))
        for (beta, gamma, Q, soft_t, soft_k) in grid:

            setattr(emb_mod, "beta", beta)
            setattr(emb_mod, "gamma", gamma)
            setattr(emb_mod, "Q", Q)
            setattr(emb_mod, "soft_t", soft_t)
            setattr(emb_mod, "soft_k", soft_k)

            try:
                Iw = embedding(img_path, WATERMARK_PATH)
            except:
                continue

            wps_clean = float(WPSNR(I_orig, Iw))
            if wps_clean < WPSNR_MIN:
                continue  # scartata

            # stampa richiesta quando supera 58
            print(f'    WPSNR OK: {wps_clean:.2f} dB con combo (beta={beta}, gamma={gamma}, Q={Q}, soft_t={soft_t}, soft_k={soft_k})')

            robustness_count = 0
            for _, afunc in ATTACKS:
                I_att = afunc(Iw)
                tmp_att = os.path.join(OUT_DIR, "tmp.png")
                cv2.imwrite(tmp_att, I_att)
                flag, _ = detection(img_path, None, tmp_att, tau=TAU)
                robustness_count += int(flag == 1)

            robustness = robustness_count / len(ATTACKS)
            results.append((wps_clean, robustness, (beta, gamma, Q, soft_t, soft_k), Iw))

        # ---- decisione finale ----
        if not results:
            print(f'wpsnr minimo non raggiunto per l\'immagine "{img_name}"')
            continue

        results.sort(key=lambda x: x[1], reverse=True)
        best = results[0]
        _, _, _, best_Iw = best

        out_path = os.path.join(WM_DIR, img_name)
        cv2.imwrite(out_path, best_Iw)
        print(f"[OK] {img_name} → salvata combo migliore (WPSNR≥58, max robustness)")

if __name__ == "__main__":
    main()
