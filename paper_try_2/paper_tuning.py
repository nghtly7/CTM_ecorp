#!/usr/bin/env python3
"""
tuning.py
Grid search tuning (SAFE mode, single-thread) for paper_embedding_v1_2.py

- MODE = FULL (AWGN, blur, sharpen, JPEG, resize, median)
- Grid: beta, gamma, Q, soft_t, soft_k as agreed
- Uses detection_nonblind from paper_detection_v1_2.py and wpsnr.py
- Saves tuning_results.csv in tuning_output/
"""

import os
import cv2
import numpy as np
import itertools
import csv
import traceback
import sys
from time import time

# ---------- Config (modify if you want) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

IMAGES_DIR = os.path.join(ROOT_DIR, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "tuning_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_OUT = os.path.join(OUTPUT_DIR, "tuning_results.csv")

# Grid (as agreed)
BETAS  = [0.4, 0.6, 0.8]
GAMMAS = [0.2, 0.4, 0.6]
QS     = [0.6, 0.8, 1.0]
SOFT_TS = [0.10, 0.15, 0.20]
SOFT_KS = [0.4, 0.6]

# Attacks (FULL): AWGN, BLUR, SHARPEN, JPEG, RESIZE, MEDIAN
AWGN_SIGMAS = [1.0, 3.0, 5.0, 8.0]
GAUSSIAN_SIGMAS = [0.8, 1.2, 1.6]
SHARP_STRENGTHS = [0.5, 1.0]
JPEG_QUALITIES = [90, 80, 70]
RESIZE_SCALES = [0.9, 0.75, 0.5]
MEDIAN_KS = [3, 5]

#! Robustness threshold (mean across attack variants & images)
ROBUSTNESS_THRESHOLD = 0.80

#! Detection threshold tau default used by detection module (it may have its own default)
TAU = 0.75

# ---------- Imports of your modules ----------
import importlib

# Fix PYTHONPATH so that detection, embedding and wpsnr can be imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # folder of tuning.py
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))  # project root
sys.path.append(ROOT_DIR)

try:
    from wpsnr import wpsnr as WPSNR
except Exception as e:
    raise RuntimeError("Could not import wpsnr from ROOT_DIR: " + str(e))

# import embedding
try:
    emb_mod = importlib.import_module("paper_embedding_v1_2")
    embedding = getattr(emb_mod, "embedding")
except Exception as e:
    raise RuntimeError("Could not import embedding from paper_embedding_v1_2: " + str(e))

# import detection
try:
    det_mod = importlib.import_module("paper_detection_v1_2")
    detection = getattr(det_mod, "detection")  # <-- Nome corretto della tua funzione
except Exception as e:
    raise RuntimeError("Could not import detection from paper_detection_v1_2: " + str(e))


# ---------- Helper attack functions ----------
rng_global = np.random.default_rng(0)

def attack_awgn(img, sigma):
    imgf = img.astype(np.float32)
    noise = rng_global.normal(0, sigma, imgf.shape).astype(np.float32)
    out = imgf + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def attack_blur(img, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

def attack_sharpen(img, strength):
    blurred = cv2.GaussianBlur(img, (3,3), sigmaX=1.0)
    out = img.astype(np.float32) + strength * (img.astype(np.float32) - blurred.astype(np.float32))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def attack_jpeg(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    dec = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
    return dec

def attack_resize(img, scale):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back

def attack_median(img, k):
    return cv2.medianBlur(img, k)

# Build attack variants list (tuples: name, function)
ATTACK_VARIANTS = []
for s in AWGN_SIGMAS:
    ATTACK_VARIANTS.append(("AWGN_sigma{:.2f}".format(s), lambda im, ss=s: attack_awgn(im, ss)))
for s in GAUSSIAN_SIGMAS:
    ATTACK_VARIANTS.append(("BLUR_sigma{:.2f}".format(s), lambda im, ss=s: attack_blur(im, ss)))
for st in SHARP_STRENGTHS:
    ATTACK_VARIANTS.append(("SHARP_str{:.2f}".format(st), lambda im, stt=st: attack_sharpen(im, stt)))
for q in JPEG_QUALITIES:
    ATTACK_VARIANTS.append(("JPEG_q{}".format(q), lambda im, qq=q: attack_jpeg(im, qq)))
for sc in RESIZE_SCALES:
    ATTACK_VARIANTS.append(("RESIZE_s{:.2f}".format(sc), lambda im, sca=sc: attack_resize(im, sca)))
for k in MEDIAN_KS:
    ATTACK_VARIANTS.append(("MEDIAN_k{}".format(k), lambda im, kk=k: attack_median(im, kk)))

# ---------- Utility ----------
def list_images(folder):
    exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif")
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]

images = list_images(IMAGES_DIR)
if len(images) == 0:
    raise RuntimeError(f"No images found in {IMAGES_DIR}")

param_grid = list(itertools.product(BETAS, GAMMAS, QS, SOFT_TS, SOFT_KS))
print("Param combos:", len(param_grid), "Images:", len(images), "Attack variants per image:", len(ATTACK_VARIANTS))

# CSV header
header = ["beta", "gamma", "Q", "soft_t", "soft_k", "wpsnr_mean", "robustness_mean", "num_images", "num_attacks", "notes"]
with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)

# Main loop (SAFE single-thread)
results = []
start_time = time()
combo_idx = 0
for (beta, gamma, Q, soft_t, soft_k) in param_grid:
    combo_idx += 1
    t0 = time()
    wpsnrs_per_image = []
    robustness_per_image = []
    notes = ""
    print(f"\n[{combo_idx}/{len(param_grid)}] beta={beta} gamma={gamma} Q={Q} soft_t={soft_t} soft_k={soft_k}")

    # Set module-level params on embedding module if present
    # We assume paper_embedding_v1_2 checks module-level variables when embedding is called without kwargs.
    try:
        setattr(emb_mod, "beta", beta)
        setattr(emb_mod, "gamma", gamma)
        setattr(emb_mod, "Q", Q)
        setattr(emb_mod, "soft_t", soft_t)
        setattr(emb_mod, "soft_k", soft_k)
    except Exception:
        pass

    # For each image
    total_present = 0
    total_tests = 0
    for img_name in images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        I = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if I is None:
            print("  ! skip, can't load:", img_path)
            continue

        # # call embedding. Try kwargs first (if function supports), otherwise rely on module globals
        # try:
        #     Iw = embedding(img_path, None, beta=beta, gamma=gamma, Q=Q, soft_t=soft_t, soft_k=soft_k)
        # except TypeError:
        #     try:
        #         # call with only path, embedding reads globals from emb_mod
        #         Iw = embedding(img_path)
        #     except Exception as e:
        #         print("  ERROR during embedding for", img_name, ":", e)
        #         traceback.print_exc()
        #         notes += f"embed_err_{img_name};"
        #         continue
        # except Exception as e:
        #     print("  ERROR during embedding for", img_name, ":", e)
        #     traceback.print_exc()
        #     notes += f"embed_err_{img_name};"
        #     continue
        
        # parametri embedding (impostati a livello modulo)
        setattr(emb_mod, "beta", beta)
        setattr(emb_mod, "gamma", gamma)
        setattr(emb_mod, "Q", Q)
        setattr(emb_mod, "soft_t", soft_t)
        setattr(emb_mod, "soft_k", soft_k)
        
        # chiamiamo embedding con watermark fisso
        Iw = embedding(img_path, "ecorp.npy")

        # compute clean WPSNR (I vs Iw) using official function
        try:
            wps_clean = float(WPSNR(I, Iw))
        except Exception as e:
            print("  WPSNR error:", e)
            wps_clean = float('nan')

        wpsnrs_per_image.append(wps_clean)

        # save temporary watermarked image (so detection can read it)
        tmp_wm_path = os.path.join(OUTPUT_DIR, f"tmp_{img_name}_wm.png")
        cv2.imwrite(tmp_wm_path, Iw)

        # apply all attack variants and test detection
        present_count = 0
        variants_tested = 0
        for (aname, afunc) in ATTACK_VARIANTS:
            try:
                I_att = afunc(Iw)
            except Exception as e:
                print("   ! attack error", aname, "->", e)
                continue

            # save attacked image temporarily
            tmp_att_path = os.path.join(OUTPUT_DIR, f"tmp_{img_name}_att_{aname}.png")
            cv2.imwrite(tmp_att_path, I_att)

            # call detection_nonblind(orig, wm, att)
            try:
                present_flag, wps_att = detection(img_path, tmp_wm_path, tmp_att_path, tau=TAU)
            except Exception as e:
                print("   ! detection error for", aname, "->", e)
                traceback.print_exc()
                present_flag = 0

            present_count += int(bool(present_flag))
            variants_tested += 1
            total_present += int(bool(present_flag))
            total_tests += 1

        # robustness for this image = fraction of attack variants where watermark present
        robustness_img = (present_count / variants_tested) if variants_tested > 0 else 0.0
        robustness_per_image.append(robustness_img)

    # aggregate across images
    wpsnr_mean = float(np.nanmean(wpsnrs_per_image)) if len(wpsnrs_per_image) > 0 else float('nan')
    robustness_mean = float(np.mean(robustness_per_image)) if len(robustness_per_image) > 0 else 0.0

    # write CSV row
    with open(CSV_OUT, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([beta, gamma, Q, soft_t, soft_k, wpsnr_mean, robustness_mean, len(wpsnrs_per_image), len(ATTACK_VARIANTS), notes])

    results.append((beta, gamma, Q, soft_t, soft_k, wpsnr_mean, robustness_mean))

    elapsed = time() - t0
    print(f"  -> wpsnr_mean={wpsnr_mean:.3f}, robustness_mean={robustness_mean:.3f} (elapsed {elapsed:.1f}s)")

# After testing all combos: sort and print top results meeting robustness threshold
results_sorted = sorted(results, key=lambda r: (r[6], r[5]), reverse=True)  # sort by robustness then wpsnr
print("\n=== TOP results (robustness desc, wpsnr desc) ===")
top_showed = 0
for r in results_sorted:
    beta, gamma, Q, soft_t, soft_k, wpsnr_mean, robustness_mean = r
    if robustness_mean >= ROBUSTNESS_THRESHOLD and not np.isnan(wpsnr_mean):
        print(f"beta={beta} gamma={gamma} Q={Q} soft_t={soft_t} soft_k={soft_k} | wpsnr={wpsnr_mean:.3f} | robustness={robustness_mean:.3f}")
        top_showed += 1
    if top_showed >= 20:
        break

print("\nTuning completed. CSV saved to:", CSV_OUT)
total_elapsed = time() - start_time
print(f"Total elapsed: {total_elapsed/60.0:.2f} minutes")
