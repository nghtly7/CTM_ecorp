#!/usr/bin/env python3
"""
tuning_per_image.py
- Tuning per IMMAGINE (non globale)
- Criterio: WPSNR-first tra le combinazioni che superano una robustezza minima
- Output:
  - tuning_output/per_image/<name>_best_params.json
  - watermarked_images/<name> (re-embed con best params)
  - tuning_output/per_image/<name>_report.csv (dettaglio combinazioni testate)

Assume:
- ecorp.npy è nella root del progetto
- embedding legge i parametri da variabili globali del modulo (beta, gamma, Q, soft_t, soft_k)
- detection signature: detection(orig, wm, att, tau) -> (present_flag, wpsnr_att)
"""

import os, sys, json, csv, itertools, traceback
import numpy as np
import cv2
from time import time
import importlib

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../paper_try_2
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))   # progetto root
sys.path.append(ROOT_DIR)

IMAGES_DIR = os.path.join(ROOT_DIR, "images")
WM_DIR     = os.path.join(ROOT_DIR, "watermarked_images")
OUT_DIR    = os.path.join(ROOT_DIR, "tuning_output", "per_image")
os.makedirs(WM_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --- GRID (stessa della FULL) ---
BETAS   = [0.4, 0.6, 0.8]
GAMMAS  = [0.2, 0.4, 0.6]
QS      = [0.6, 0.8, 1.0]
SOFT_TS = [0.10, 0.15, 0.20]
SOFT_KS = [0.4, 0.6, 0.8]

ROBUSTNESS_MIN = 0.80    # filtro combinazioni scarse
TAU = 0.75               # soglia detection

# --- IMPORT MODULI TUOI ---
from wpsnr import wpsnr as WPSNR
emb_mod = importlib.import_module("paper_embedding_v1_2")
embedding = getattr(emb_mod, "embedding")
det_mod = importlib.import_module("paper_detection_v1_2")
detection = getattr(det_mod, "detection")  # detection(orig, wm, att, tau) -> (present_flag, wpsnr_att)

# --- ATTACCHI (FULL) ---
rng_global = np.random.default_rng(0)
def attack_awgn(img, sigma):
    imgf = img.astype(np.float32)
    noise = rng_global.normal(0, sigma, imgf.shape).astype(np.float32)
    out = np.clip(imgf + noise, 0, 255).astype(np.uint8)
    return out

def attack_blur(img, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

def attack_sharpen(img, strength):
    blurred = cv2.GaussianBlur(img, (3,3), sigmaX=1.0)
    out = img.astype(np.float32) + strength * (img.astype(np.float32) - blurred.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)

def attack_jpeg(img, quality):
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return dec

def attack_resize(img, scale):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    back  = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back

def attack_median(img, k):
    return cv2.medianBlur(img, k)

AWGN_SIGMAS      = [1.0, 3.0, 5.0, 8.0]
GAUSSIAN_SIGMAS  = [0.8, 1.2, 1.6]
SHARP_STRENGTHS  = [0.5, 1.0]
JPEG_QUALITIES   = [90, 80, 70]
RESIZE_SCALES    = [0.9, 0.75, 0.5]
MEDIAN_KS        = [3, 5]

ATTACKS = []
for s in AWGN_SIGMAS:
    ATTACKS.append((f"AWGN_{s:.2f}", lambda im, ss=s: attack_awgn(im, ss)))
for s in GAUSSIAN_SIGMAS:
    ATTACKS.append((f"BLUR_{s:.2f}", lambda im, ss=s: attack_blur(im, ss)))
for st in SHARP_STRENGTHS:
    ATTACKS.append((f"SHARP_{st:.2f}", lambda im, stt=st: attack_sharpen(im, stt)))
for q in JPEG_QUALITIES:
    ATTACKS.append((f"JPEG_{q}", lambda im, qq=q: attack_jpeg(im, qq)))
for sc in RESIZE_SCALES:
    ATTACKS.append((f"RESIZE_{sc:.2f}", lambda im, sca=sc: attack_resize(im, sca)))
for k in MEDIAN_KS:
    ATTACKS.append((f"MEDIAN_{k}", lambda im, kk=k: attack_median(im, kk)))

def list_images(folder):
    exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif")
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]

def eval_combo_on_image(img_path, params):
    """Embed + attacchi + detection su una singola immagine. Ritorna (wpsnr_clean, robustness, notes)."""
    beta, gamma, Q, soft_t, soft_k = params
    # set parametri globali
    setattr(emb_mod, "beta", beta)
    setattr(emb_mod, "gamma", gamma)
    setattr(emb_mod, "Q", Q)
    setattr(emb_mod, "soft_t", soft_t)
    setattr(emb_mod, "soft_k", soft_k)

    I = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if I is None:
        return (np.nan, 0.0, "load_err")

    # embed (usa watermark fisso ecorp.npy nella root)
    try:
        Iw = embedding(img_path, "ecorp.npy")
    except Exception as e:
        return (np.nan, 0.0, f"embed_err:{e}")

    # WPSNR clean
    try:
        wps_clean = float(WPSNR(I, Iw))
    except Exception as e:
        wps_clean = float('nan')

    # detection dopo attacchi
    wm_tmp_path  = os.path.join(OUT_DIR, f"tmp_{os.path.basename(img_path)}_wm.png")
    cv2.imwrite(wm_tmp_path, Iw)
    present = 0; tested = 0
    for (name, afunc) in ATTACKS:
        try:
            I_att = afunc(Iw)
        except Exception as e:
            continue
        att_tmp = os.path.join(OUT_DIR, f"tmp_{os.path.basename(img_path)}_{name}.png")
        cv2.imwrite(att_tmp, I_att)
        try:
            flag, wps_att = detection(img_path, wm_tmp_path, att_tmp, tau=TAU)
        except Exception:
            flag = 0
        present += int(bool(flag))
        tested  += 1

    robustness = (present/tested) if tested > 0 else 0.0
    return (wps_clean, robustness, "")

def main():
    images = list_images(IMAGES_DIR)
    if not images:
        print(f"Nessuna immagine in {IMAGES_DIR}")
        return

    print(f"Per-image tuning su {len(images)} immagini | combinazioni={len(BETAS)*len(GAMMAS)*len(QS)*len(SOFT_TS)*len(SOFT_KS)} | attacchi={len(ATTACKS)}")

    grid = list(itertools.product(BETAS, GAMMAS, QS, SOFT_TS, SOFT_KS))

    for img_name in images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        rep_csv  = os.path.join(OUT_DIR, f"{os.path.splitext(img_name)[0]}_report.csv")
        best_json= os.path.join(OUT_DIR, f"{os.path.splitext(img_name)[0]}_best_params.json")

        print(f"\n🔹 Image: {img_name}")
        # report csv per l'immagine
        with open(rep_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["beta","gamma","Q","soft_t","soft_k","wpsnr_clean","robustness"])

            best_pass = None  # migliore tra quelle che superano robustezza min
            best_any  = None  # migliore in assoluto (per fallback se nessuna supera)

            for (beta, gamma, Q, soft_t, soft_k) in grid:
                wps_clean, rob, note = eval_combo_on_image(img_path, (beta, gamma, Q, soft_t, soft_k))
                w.writerow([beta, gamma, Q, soft_t, soft_k, wps_clean, rob])

                # aggiorna best_any (per fallback): robustezza -> wpsnr
                if best_any is None or (rob > best_any[1] or (rob == best_any[1] and (wps_clean > best_any[0]))):
                    best_any = (wps_clean, rob, (beta, gamma, Q, soft_t, soft_k))

                # se supera soglia, candida per best_pass (WPSNR-first)
                if rob >= ROBUSTNESS_MIN:
                    if (best_pass is None) or (wps_clean > best_pass[0]):
                        best_pass = (wps_clean, rob, (beta, gamma, Q, soft_t, soft_k))

        # scegli combo finale per l’immagine
        if best_pass is not None:
            chosen = best_pass
            reason = "WPSNR-first tra comb. robuste"
        else:
            chosen = best_any
            reason = "fallback: best robustness, poi WPSNR"

        wps_clean, rob, (beta, gamma, Q, soft_t, soft_k) = chosen
        print(f"   → BEST params: beta={beta} gamma={gamma} Q={Q} soft_t={soft_t} soft_k={soft_k} | WPSNR={wps_clean:.2f} dB | Robustezza={rob:.2f}  [{reason}]")

        # salva JSON con parametri scelti
        with open(best_json, "w") as f:
            json.dump({
                "image": img_name,
                "beta": beta, "gamma": gamma, "Q": Q, "soft_t": soft_t, "soft_k": soft_k,
                "wpsnr_clean": wps_clean, "robustness": rob, "reason": reason
            }, f, indent=2)

        # re-embed definitivo e salva in watermarked_images/
        setattr(emb_mod, "beta", beta)
        setattr(emb_mod, "gamma", gamma)
        setattr(emb_mod, "Q", Q)
        setattr(emb_mod, "soft_t", soft_t)
        setattr(emb_mod, "soft_k", soft_k)

        Iw_final = embedding(img_path, "ecorp.npy")
        out_final = os.path.join(WM_DIR, img_name)
        cv2.imwrite(out_final, Iw_final)
        print(f"   → Salvata watermarked finale: {out_final}")

    print("\n✅ Per-image tuning completato. Risultati in:", OUT_DIR)

if __name__ == "__main__":
    main()
