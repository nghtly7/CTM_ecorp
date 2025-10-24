import os
import cv2
import numpy as np
import sys

# Aggiunge la cartella corrente (CTM_ecorp) al path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Aggiunge anche le sottocartelle dove hai i moduli
sys.path.append(os.path.join(BASE_DIR, "embedding"))
sys.path.append(os.path.join(BASE_DIR, "detection"))
from embedding.Ecorp_strategy_with_ranking import embedding
from detection.Ecorp_detection_with_ranking import detection

# --- CONFIGURAZIONE ---
INPUT_DIR = "sample-images"
OUTPUT_DIR = "watermarked"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Funzione per WPSNR ---
def wpsnr(img1, img2):
    """Calcola il WPSNR tra due immagini in scala di grigi o RGB."""
    if img1.shape != img2.shape:
        raise ValueError("Le immagini devono avere la stessa dimensione")
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    mse = np.mean((img1_f - img2_f) ** 2)
    if mse == 0:
        return float('inf')
    max_i = 255.0
    return 10 * np.log10((max_i ** 2) / mse)

# --- Main Loop ---
results = []
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(".bmp"):
        continue

    path_in = os.path.join(INPUT_DIR, fname)
    path_out = os.path.join(OUTPUT_DIR, f"wm_{fname}")

    print(f"\n▶ Elaboro: {fname}")

    try:
        # 1️⃣ Embedding del watermark
        I_wm_u8, stats = embedding(path_in, None, None, None, None)

        # 2️⃣ Salva immagine watermarkata
        cv2.imwrite(path_out, I_wm_u8)

        # 3️⃣ Calcolo WPSNR
        orig = cv2.imread(path_in)
        if orig is None:
            raise RuntimeError(f"Impossibile leggere {path_in}")
        wpsnr_val = wpsnr(orig, I_wm_u8)

        # 4️⃣ Detection (verifica watermark)
        detected, conf = detection(path_out)

        # 5️⃣ Aggiungi ai risultati
        results.append({
            "file": fname,
            "wpsnr": wpsnr_val,
            "detected": detected,
            "confidence": conf,
        })

        print(f"   ✅ WPSNR: {wpsnr_val:.2f} dB | Watermark rilevato: {detected} (conf={conf:.3f})")

    except Exception as e:
        print(f"   ❌ Errore su {fname}: {e}")
        results.append({"file": fname, "wpsnr": None, "detected": False, "confidence": 0.0})

# --- Riepilogo finale ---
print("\n=== RISULTATI FINALI ===")
for r in results:
    print(f"{r['file']:25s} | WPSNR = {r['wpsnr'] if r['wpsnr'] else 'N/A':>7} | Detected = {r['detected']} | conf = {r['confidence']:.3f}")

# (opzionale) salva CSV
import pandas as pd
pd.DataFrame(results).to_csv("batch_results.csv", index=False)
print("\n💾 File dei risultati salvato: batch_results.csv")
