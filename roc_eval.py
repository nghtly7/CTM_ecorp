#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roc_eval.py — ROC per il tuo sistema di watermarking con cartella cover hard-coded.

- Usa direttamente ecorp_embedding.py ed ecorp_detection.py (stessi nomi del tuo repo).
- Genera campioni POSITIVI (cover -> embed -> attacco) e NEGATIVI (cover -> attacco, senza watermark).
- Calcola uno score CONTINUO con coseno su bit bipolari (0/1 -> -1/+1), poi traccia ROC + AUC.
- Salva:
    - plot PNG della ROC
    - file .npz con scores/labels/fpr/tpr/thresholds/auc
- La cartella delle cover è definita in COVERS_DIR (qui sotto).
"""

import os
import cv2
import argparse
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import roc_curve, auc

COVERS_DIR = "input_images/sample-images"  # cartella di default per immagini cover

# importa le funzioni dal tuo codice
from final_strategy.ecorp_embedding import embedding as embed_mod
from final_strategy.ecorp_detection import detection as detect_mod

# ---- Attacchi disponibili ----
def attack_identity(img):
    return img.copy()

def attack_jpeg(img, quality=75):
    encok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not encok:
        raise RuntimeError("JPEG encode fallita")
    out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out

def attack_awgn(img, sigma=5.0):
    noise = np.random.normal(0.0, float(sigma), img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def attack_blur(img, ksize=3):
    k = int(ksize)
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(img, (k, k), 0)

# ---- Similarità/Score ----
def bits_to_bipolar(bits01):
    return 2.0 * bits01.astype(np.float32) - 1.0  # {0,1} -> {-1,+1}

def bipolar_cosine_score(b1, b2):
    v1 = bits_to_bipolar(b1)
    v2 = bits_to_bipolar(b2)
    num = float(np.dot(v1, v2))
    den = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return num / den  # ~0 per caso, >0 quando c'è concordanza

# ---- Utilità ----
def read_gray_512(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f"Impossibile leggere: {path}")
    if im.shape != (512, 512):
        im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    return im

def list_cover_files(folder):
    pats = ('*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff')
    files = []
    for p in pats:
        files.extend(glob(os.path.join(folder, p)))
    return sorted(files)

# ---- Core ROC ----
def evaluate_roc_hardcoded(outdir='./roc_out',
                           attack='identity',
                           param=None,
                           max_images=None,
                           seed=42):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    files = list_cover_files(COVERS_DIR)
    if max_images:
        files = files[:max_images]
    if not files:
        raise RuntimeError(f"Nessuna immagine trovata in: {COVERS_DIR}")

    # pick attacco
    if attack == 'identity':
        attack_fn = lambda im: attack_identity(im)
    elif attack == 'jpeg':
        q = int(param) if param is not None else 75
        attack_fn = lambda im: attack_jpeg(im, q)
    elif attack == 'awgn':
        sigma = float(param) if param is not None else 5.0
        attack_fn = lambda im: attack_awgn(im, sigma)
    elif attack == 'blur':
        k = int(param) if param is not None else 3
        attack_fn = lambda im: attack_blur(im, k)
    else:
        raise ValueError(f"Attack non supportato: {attack}")

    tmpdir = tempfile.mkdtemp(prefix='roc_tmp_')
    print(f"[INFO] Covers: {len(files)} | Attack={attack} param={param} | Temp={tmpdir}")

    scores, labels, wpsnrs = [], [], []

    for i, cover_path in enumerate(files):
        try:
            # Carica cover (per eventuale negativo e per extraction reference)
            I_orig = read_gray_512(cover_path)

            # === POSITIVO ===
            # 1) embed su cover
            I_w = embed_mod.embedding(cover_path)  # deve restituire uint8 512x512
            wm_path = os.path.join(tmpdir, f"wm_{i}.png")
            cv2.imwrite(wm_path, I_w)

            # 2) attacco su watermarked
            I_att_w = attack_fn(I_w)
            wm_att_path = os.path.join(tmpdir, f"wm_att_{i}.png")
            cv2.imwrite(wm_att_path, I_att_w)

            # 3) extraction -> bit
            bits_ref = detect_mod.extraction(cover_path, wm_path)
            bits_att = detect_mod.extraction(cover_path, wm_att_path)

            # 4) score continuo (coseno bipolare)
            s_pos = bipolar_cosine_score(bits_ref, bits_att)
            scores.append(s_pos); labels.append(1)

            # WPSNR (diagnostica)
            try:
                wpsnrs.append(detect_mod.wpsnr(I_w, I_att_w))
            except Exception:
                wpsnrs.append(np.nan)

            # === NEGATIVO === (stessa cover senza watermark, ma attaccata)
            I_att_o = attack_fn(I_orig)
            orig_att_path = os.path.join(tmpdir, f"orig_att_{i}.png")
            cv2.imwrite(orig_att_path, I_att_o)

            bits_neg = detect_mod.extraction(cover_path, orig_att_path)
            s_neg = bipolar_cosine_score(bits_ref, bits_neg)
            scores.append(s_neg); labels.append(0)

            # altra WPSNR di riferimento (non fondamentale)
            try:
                wpsnrs.append(detect_mod.wpsnr(I_w, I_att_o))
            except Exception:
                wpsnrs.append(np.nan)

            if (i+1) % 10 == 0:
                print(f"[INFO] processate {i+1}/{len(files)} cover")

        except Exception as e:
            print(f"[WARN] {cover_path}: {e}")

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    # ROC/AUC
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC — attack={attack}, param={param}')
    plt.legend(loc='lower right')
    out_png = os.path.join(outdir, f'roc_{attack}_{param}.png')
    plt.savefig(out_png, dpi=200)
    plt.close()

    # Salva numeri
    np.savez_compressed(os.path.join(outdir, f'scores_{attack}_{param}.npz'),
                        scores=scores, labels=labels, fpr=fpr, tpr=tpr, thr=thr,
                        auc=roc_auc, wpsnrs=np.asarray(wpsnrs, dtype=np.float32))

    print(f"[DONE] ROC salvata in: {out_png} | AUC={roc_auc:.4f}")
    return fpr, tpr, thr, roc_auc


# ---- CLI (attacco/param opzionali) ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ROC evaluation (covers dir hard-coded).")
    ap.add_argument('--outdir', default='./roc_out', help='cartella output')
    ap.add_argument('--attack', default='identity', choices=['identity','jpeg','awgn','blur'])
    ap.add_argument('--param', default=None, help='parametro attacco (q, sigma, ksize)')
    ap.add_argument('--max', type=int, default=None, help='usa al massimo N immagini')
    ap.add_argument('--seed', type=int, default=42, help='seed RNG')
    args = ap.parse_args()

    evaluate_roc_hardcoded(outdir=args.outdir,
                           attack=args.attack,
                           param=args.param,
                           max_images=args.max,
                           seed=args.seed)
