#!/usr/bin/env python3
"""
embedding_grid_search.py
Full grid search over embedding parameters to maximize worst-case similarity across attacks,
subject to WPSNR >= 54.0 dB.

Usage:
    python embedding_grid_search.py

Outputs:
 - grid_results.csv
 - best_config.json
 - temp files in grid_tmp/
"""

#! da modificare

import os
import sys
import cv2
import json
import time
import shutil
import hashlib
import itertools
import numpy as np
import pywt
import traceback
import multiprocessing
import csv
#from wpsnr import wpsnr as wpsnr_fn

ROOT = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(ROOT, "images")
WATERMARK_FILE = os.path.join(ROOT, "ecorp.npy")  # change if needed

TMP_DIR = os.path.join(ROOT, "grid_tmp")
WM_DIR = os.path.join(TMP_DIR, "wm")
ATT_DIR = os.path.join(TMP_DIR, "attacked")
OUT_CSV = os.path.join(ROOT, "grid_results.csv")
OUT_BEST = os.path.join(ROOT, "best_config.json")

os.makedirs(WM_DIR, exist_ok=True)
os.makedirs(ATT_DIR, exist_ok=True)

# ---- import detection helpers (extraction, similarity, wpsnr)
try:
    import detection as detmod
except Exception:
    try:
        from paper_try_2 import detection as detmod
    except Exception:
        raise ImportError("Impossibile importare paper_detection_v1; assicurati sia nel PYTHONPATH.")

extraction_fn = detmod.extraction
similarity_fn = detmod.similarity
wpsnr_fn = detmod.wpsnr
# ---- ATTACKS IMPLEMENTED LOCALLY (deterministic)
def attack_awgn(img, sigma):
    noise = np.random.default_rng().normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def attack_blur(img, ksize):
    if ksize % 2 == 0: ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0).astype(np.uint8)

def attack_jpeg(img, q):
    # compress and decompress via cv2.imencode
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    if not result:
        raise RuntimeError("JPEG encode failed")
    dec = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
    return dec

def attack_resize(img, scale):
    h,w = img.shape
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # resize back to original
    back = cv2.resize(small, (w,h), interpolation=cv2.INTER_LINEAR)
    return back

def attack_median(img, k):
    if k % 2 == 0: k += 1
    return cv2.medianBlur(img, k)

def attack_sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    out = cv2.filter2D(img.astype(np.float32), -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)

# list of attacks and parameter sets used by grid evaluation
ATTACKS = [
    ("awgn", [2.0, 4.0]),
    ("blur", [3, 5]),
    ("jpeg", [30, 50]),
    ("resize", [0.90, 1.10]),
    ("median", [3]),
    ("sharpen", [None]),
]

# ---- Parameter grid (as discussed)
GAMMAS = [0.45, 0.50, 0.55, 0.60]
EQ_METHODS = [0, 1, 2]   # 0=off, 1=1/sigma, 2=1/(sigma^2)
L3_WEIGHTS = [1.00, 1.10, 1.20]
L2_WEIGHTS = [0.55, 0.65, 0.75]
HVS_SCALES = [1.00, 1.07, 1.15]

grid = list(itertools.product(GAMMAS, EQ_METHODS, L3_WEIGHTS, L2_WEIGHTS, HVS_SCALES))
print(f"[GRID] {len(grid)} configurations")

# ---- Embedding implementation (parametrizable version of your embedding)
def embedding_param(input1, input2, gamma=0.55, eq_method=1, l3_weight=1.0, l2_weight=0.65, hvs_scale=1.07, target_wpsnr=54.0):
    """
    Parametric embedding that mirrors your embedding(input1,input2) but allows runtime params.
    Returns (Iw_uint8, achieved_wpsnr)
    """
    # internal constants similar to your implementation
    WAVELET="db2"; LEVELS=3; FIXED_SEED=42; EPS=1e-8
    # build HVS weights base and apply hvs_scale to mid-high entries (the last two)
    base_hvs = np.array([0.55, 0.60, 0.95, 1.05, 1.05, 1.15, 1.15], dtype=np.float32)
    HVS_WEIGHTS = base_hvs.copy()
    # scale the mid-high coefficients (positions 4,5,6 indices 3..6) mildly
    HVS_WEIGHTS[4:] *= hvs_scale

    BAND_WEIGHTS = {("HL",3): l3_weight, ("LH",3): l3_weight, ("HL",2): l2_weight, ("LH",2): l2_weight}

    # read image and watermark bits
    I = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if I is None:
        raise FileNotFoundError(input1)
    I_f = I.astype(np.float32)

    if input2 is None or not os.path.exists(input2):
        rngtmp = np.random.default_rng(FIXED_SEED)
        Wbits = rngtmp.integers(0,2,size=1024,dtype=np.uint8)
    else:
        Wbits = np.load(input2).astype(np.uint8).flatten()
        if Wbits.size != 1024:
            raise ValueError("watermark must be 1024 bits")

    # DWT
    coeffs = pywt.wavedec2(I_f, wavelet=WAVELET, level=LEVELS)
    (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs[1], coeffs[2], coeffs[3]
    bands = [("HL",3,HL3), ("LH",3,LH3), ("HL",2,HL2), ("LH",2,LH2)]

    MASK_IDX = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]
    rng = np.random.default_rng(FIXED_SEED)
    PN0 = rng.standard_normal(len(MASK_IDX)).astype(np.float32)
    PN1 = rng.standard_normal(len(MASK_IDX)).astype(np.float32)
    PN0 = PN0 / (np.linalg.norm(PN0)+EPS)
    PN1 = PN1 - (PN0 @ PN1) * PN0
    PN1 = PN1 / (np.linalg.norm(PN1)+EPS)

    # helper to embed using a global kappa
    def _embed_with_kappa(kappa):
        HL3c, LH3c, HL2c, LH2c = [b[2].copy() for b in bands]
        band_map = {("HL",3):HL3c, ("LH",3):LH3c, ("HL",2):HL2c, ("LH",2):LH2c}
        idx = 0
        for (name, level, B) in [("HL",3,HL3c),("LH",3,LH3c),("HL",2,HL2c),("LH",2,LH2c)]:
            r_band = BAND_WEIGHTS[(name, level)]
            for by in range(0,64,4):
                for bx in range(0,64,4):
                    block = B[by:by+4, bx:bx+4].astype(np.float32, copy=True)
                    C = cv2.dct(block)
                    mb = np.array([C[u,v] for (u,v) in MASK_IDX], dtype=np.float32)
                    mb_std = float(np.std(mb)) + EPS
                    alpha_base = kappa * r_band * (mb_std**gamma)
                    # equalization
                    if eq_method == 0:
                        eq = 1.0
                    elif eq_method == 1:
                        eq = 1.0 / (mb_std)
                    else:
                        eq = 1.0 / (mb_std**2)
                    bit = int(Wbits[idx])
                    PN = PN1 if bit==1 else PN0
                    for k,(u,v) in enumerate(MASK_IDX):
                        C[u,v] += alpha_base * HVS_WEIGHTS[k] * PN[k] * eq
                    B[by:by+4, bx:bx+4] = cv2.idct(C)
                    idx += 1
            band_map[(name, level)] = B
        new_coeffs = list(coeffs)
        new_coeffs[1] = (band_map[("LH",3)], band_map[("HL",3)], HH3)
        new_coeffs[2] = (band_map[("LH",2)], band_map[("HL",2)], HH2)
        new_coeffs[3] = (LH1, HL1, HH1)
        Iw = pywt.waverec2(new_coeffs, wavelet=WAVELET)
        Iw = np.clip(Iw,0,255).astype(np.float32)
        return Iw

    # wrapper wpsnr
    def _wpsnr_wrapper(A, B):
        return float(wpsnr_fn(A.astype(np.uint8), B.astype(np.uint8)))

    # find kappa: doubling then bisection (like before)
    k_lo, k_hi = 0.0, 1.0
    Iw_hi = _embed_with_kappa(k_hi)
    W_hi = _wpsnr_wrapper(I_f, Iw_hi)
    tries = 0
    while W_hi >= target_wpsnr and tries < 12:
        k_hi *= 2.0
        Iw_hi = _embed_with_kappa(k_hi)
        W_hi = _wpsnr_wrapper(I_f, Iw_hi)
        tries += 1
    best_Iw = None
    best_k = k_lo
    for _ in range(10):
        k_mid = 0.5*(k_lo + k_hi)
        Iw_mid = _embed_with_kappa(k_mid)
        W_mid = _wpsnr_wrapper(I_f, Iw_mid)
        if W_mid >= target_wpsnr:
            best_Iw = Iw_mid
            best_k = k_mid
            k_lo = k_mid
        else:
            k_hi = k_mid
        if abs(k_hi - k_lo) < 1e-3:
            break
    if best_Iw is None:
        best_Iw = Iw_hi if W_hi >= target_wpsnr else _embed_with_kappa(k_lo)
    return np.clip(best_Iw,0,255).astype(np.uint8), _wpsnr_wrapper(I_f, best_Iw)


# ---- iterate over grid and dataset
images = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".bmp")]
if len(images) == 0:
    raise RuntimeError("Nessuna immagine .bmp in sample-images/")

# results CSV header
header = ["gamma","eq_method","l3_weight","l2_weight","hvs_scale",
          "worst_similarity","avg_similarity","avg_wpsnr","time_s","notes"]

with open(OUT_CSV, "w", newline="") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(header)

best_overall = None

start_all = time.time()
for (gamma, eq_m, l3w, l2w, hvs_s) in grid:
    cfg_start = time.time()
    config = dict(gamma=gamma, eq_method=eq_m, l3_weight=l3w, l2_weight=l2w, hvs_scale=hvs_s)
    sims = []
    wpsnrs = []
    notes = ""
    try:
        for imgname in images:
            input_path = os.path.join(SAMPLE_DIR, imgname)
            # embed with params
            Iw, achieved_wpsnr = embedding_param(input_path, WATERMARK_FILE,
                                                 gamma=gamma, eq_method=eq_m,
                                                 l3_weight=l3w, l2_weight=l2w,
                                                 hvs_scale=hvs_s, target_wpsnr=54.0)
            # save watermarked
            basename = os.path.splitext(imgname)[0]
            wm_path = os.path.join(WM_DIR, f"{basename}_wm_g{gamma}_e{eq_m}_l3{l3w}_l2{l2w}_hvs{hvs_s}.bmp")
            cv2.imwrite(wm_path, Iw)
            wpsnrs.append(float(achieved_wpsnr))

            # for each attack & its params apply and compute similarity
            for (attack_name, param_list) in ATTACKS:
                for p in param_list:
                    if attack_name == "awgn":
                        att = attack_awgn(Iw, p)
                    elif attack_name == "blur":
                        att = attack_blur(Iw, p)
                    elif attack_name == "jpeg":
                        att = attack_jpeg(Iw, p)
                    elif attack_name == "resize":
                        att = attack_resize(Iw, p)
                    elif attack_name == "median":
                        att = attack_median(Iw, p)
                    elif attack_name == "sharpen":
                        att = attack_sharpen(Iw)
                    else:
                        raise ValueError("Unknown attack " + attack_name)

                    # save attacked to disk (detection.extraction expects path)
                    attack_tag = f"{attack_name}_{p}"
                    att_path = os.path.join(ATT_DIR, f"{basename}_att_{attack_tag}_g{gamma}_e{eq_m}_l3{l3w}_l2{l2w}_hvs{hvs_s}.bmp")
                    cv2.imwrite(att_path, att)

                    # extract watermarks from (original, watermarked) and (original, attacked)
                    try:
                        w_orig = extraction_fn(input_path, wm_path)
                        w_att  = extraction_fn(input_path, att_path)
                    except Exception as ex:
                        # extraction failed for this attacked image — register sim= -inf
                        print(f"[WARN] extraction failed: {ex}")
                        sims.append(-1.0)
                        continue

                    sim = float(similarity_fn(w_orig, w_att))
                    sims.append(sim)

        # compute aggregate stats across ALL images & attacks
        if len(sims) == 0:
            worst_sim = -1.0
            avg_sim = -1.0
        else:
            worst_sim = float(np.min(sims))
            avg_sim = float(np.mean(sims))
        avg_wpsnr = float(np.mean(wpsnrs)) if wpsnrs else 0.0

        elapsed = time.time() - cfg_start
        with open(OUT_CSV, "a", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow([gamma, eq_m, l3w, l2w, hvs_s, worst_sim, avg_sim, avg_wpsnr, round(elapsed,2), notes])

        # update best (maximize worst_sim)
        if best_overall is None or worst_sim > best_overall["worst_similarity"]:
            best_overall = dict(config=config, worst_similarity=worst_sim, avg_similarity=avg_sim, avg_wpsnr=avg_wpsnr)

        print(f"[GRID] done cfg gamma={gamma} eq={eq_m} l3={l3w} l2={l2w} hvs={hvs_s} -> worst_sim={worst_sim:.4f} avg_sim={avg_sim:.4f} avg_wpsnr={avg_wpsnr:.2f} ({round(elapsed,1)}s)")

    except Exception as ex:
        traceback.print_exc()
        print("[ERROR] configuration failed:", config)
        with open(OUT_CSV, "a", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow([gamma, eq_m, l3w, l2w, hvs_s, -999, -999, -999, 0, "error"])

# write best config json
if best_overall:
    with open(OUT_BEST, "w") as f:
        json.dump(best_overall, f, indent=2)
    print("[DONE] Best config:", best_overall)

print("[ALL DONE] Total time:", round(time.time() - start_all, 1), "s")
