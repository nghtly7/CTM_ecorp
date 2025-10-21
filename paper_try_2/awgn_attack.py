import os
import cv2
import numpy as np
from typing import List, Union

# --------------------
# Utility: aggiunge AWGN a un'immagine (grayscale)
# --------------------
def add_awgn(img: np.ndarray, sigma: float, seed: int = None) -> np.ndarray:
    """
    Aggiunge rumore gaussiano (AWGN) all'immagine in input.
    img    : numpy array grayscale float32 o uint8
    sigma  : deviazione standard del rumore (in intensità pixel)
    seed   : seme RNG opzionale per riproducibilità
    ritorna: immagine rumorosa (uint8, clipped 0-255)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # converti a float32 per il calcolo
    img_f = img.astype(np.float32)

    noise = rng.normal(loc=0.0, scale=sigma, size=img_f.shape).astype(np.float32)
    noisy = img_f + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# --------------------
# API richiesta dal bando: attacks(input1, attack_name, param_array)
# qui supportiamo 'awgn' (o ['awgn'])
# --------------------
def attacks(input1: str, attack_name: Union[str, List[str]], param_array: dict):
    """
    Applica attacchi al file input1.
    - input1: path dell'immagine watermarked (string)
    - attack_name: 'awgn' oppure list contenente 'awgn'
    - param_array: dizionario con parametri:
        {
          "sigma_start": float,
          "sigma_end": float,
          "n_steps": int,
          "seed": int (opzionale),
          "out_dir": str (opzionale)
        }
    Ritorna la lista dei path dei file generati (in ordine di sigma crescente).
    """
    if isinstance(attack_name, list):
        names = attack_name
    else:
        names = [attack_name]

    # per ora supportiamo solo AWGN
    if 'awgn' not in [n.lower() for n in names]:
        raise ValueError("Questo script supporta solo 'awgn' come attack_name.")

    # Parametri default
    sigma_start = float(param_array.get("sigma_start", 1.0))
    sigma_end   = float(param_array.get("sigma_end", 25.0))
    n_steps     = int(param_array.get("n_steps", 10))
    seed        = param_array.get("seed", None)
    out_dir     = param_array.get("out_dir", "attacked_images_awgn")

    # Crea cartella output se non esiste
    os.makedirs(out_dir, exist_ok=True)

    # Carica immagine
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {input1}")

    # Serie di sigma
    if n_steps <= 1:
        sigmas = np.array([sigma_end], dtype=float)
    else:
        sigmas = np.linspace(sigma_start, sigma_end, n_steps)

    out_paths = []
    base_name = os.path.splitext(os.path.basename(input1))[0]

    for i, s in enumerate(sigmas):
        # usa seme diverso per ogni livello se seed fornito
        cur_seed = None if seed is None else int(seed) + i
        noisy = add_awgn(img, sigma=float(s), seed=cur_seed)

        # filename indicativo: base_awgn_sigmaXX.png
        fname = f"{base_name}_awgn_sigma{float(s):.2f}.png"
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, noisy)
        out_paths.append(out_path)

    return out_paths

