# attacks_awgn_sweep.py
"""
attacks_awgn_sweep.py

Applicare una sweep progressiva AWGN su tutti i file in una cartella (es. watermarked_images/)
e salvare gli attaccati in attacked_images_sweep/. Produce anche un CSV di log.

Interfaccia principale:
    processed = batch_awgn_sweep(watermarked_dir, attacked_dir, std_list, seed, csv_out)

Restituisce la lista di tuple (orig_watermarked_path, attacked_path, std).
"""
import os
import csv
import math
import cv2
import numpy as np
from typing import List, Tuple, Optional

def _awgn_array(img_arr: np.ndarray, std: float, seed: Optional[int] = None) -> np.ndarray:
    """Aggiunge AWGN a un array immagine (uint8) e ritorna uint8."""
    if seed is not None:
        rng = np.random.RandomState(int(seed))
        noise = rng.normal(0.0, float(std), img_arr.shape)
    else:
        noise = np.random.normal(0.0, float(std), img_arr.shape)
    attacked = img_arr.astype(np.float64) + noise
    attacked = np.clip(attacked, 0, 255).astype(np.uint8)
    return attacked

def _safe_write(path: str, arr: np.ndarray) -> bool:
    """Scrive immagine su disco; ritorna True se OK."""
    try:
        # Assicuriamoci che la directory esista
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        return cv2.imwrite(path, arr)
    except Exception:
        return False

def batch_awgn_sweep(watermarked_dir: str,
                     attacked_dir: str,
                     std_list: List[float],
                     seed: Optional[int] = None,
                     csv_out: str = "sweep_results_awgn.csv",
                     extensions: List[str] = ['.bmp', '.png', '.jpg', '.jpeg']) -> List[Tuple[str, str, float]]:
    """
    Applica AWGN progressivo a tutti i file in watermarked_dir.
    - watermarked_dir: cartella con file watermarked_*.bmp (o altro formato)
    - attacked_dir: cartella di destinazione per attaccati
    - std_list: lista di deviazioni standard da applicare (es. [0.05,0.1,0.2,0.5,1,2,3])
    - seed: seed (opzionale) per riproducibilità (stessa seed applicata a tutte le immagini per uno std)
    - csv_out: percorso CSV di log
    Ritorna lista di (wm_path, attacked_path, std)
    """
    if not os.path.exists(watermarked_dir):
        raise FileNotFoundError(f"Cartella watermarked non trovata: {watermarked_dir}")
    os.makedirs(attacked_dir, exist_ok=True)

    all_files = sorted([f for f in os.listdir(watermarked_dir) if any(f.lower().endswith(ext) for ext in extensions)])
    processed = []

    # CSV header
    with open(csv_out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["watermarked_file", "attacked_file", "std", "seed", "saved"])

        for fname in all_files:
            wm_path = os.path.join(watermarked_dir, fname)

            # carica immagine once (in grayscale se single-channel, else keep channels)
            arr = cv2.imread(wm_path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                print(f"  ⚠ Impossibile leggere {wm_path} — skip")
                continue

            for std in std_list:
                # nome chiaro: attacked_std{std}_{origfilename}
                std_tag = f"{std:.3f}".replace('.', 'p')
                attacked_fname = f"attacked_std{std_tag}_{fname}"
                attacked_path = os.path.join(attacked_dir, attacked_fname)

                # calcola attacked e salva
                attacked_arr = _awgn_array(arr, std, seed)
                saved = _safe_write(attacked_path, attacked_arr)
                writer.writerow([wm_path, attacked_path, std, seed if seed is not None else "", "1" if saved else "0"])
                csvfile.flush()

                processed.append((wm_path, attacked_path, float(std)))

    return processed

# Quick demo (se esegui il file direttamente)
if __name__ == "__main__":
    wm_dir = "watermarked_images"
    out_dir = "attacked_images_sweep"
    # lista progressiva da molto debole a forte (modifica a piacere)
    stds = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    s = 42
    print(f"Applying AWGN sweep: {stds} on files in {wm_dir} -> {out_dir}")
    processed = batch_awgn_sweep(wm_dir, out_dir, stds, seed=s, csv_out="sweep_results_awgn.csv")
    print(f"Processed {len(processed)} attacked images (saved in {out_dir}). CSV: sweep_results_awgn.csv")
