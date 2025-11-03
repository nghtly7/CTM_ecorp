# attacks/median_attack.py
import os
import cv2
import numpy as np
from typing import List, Union

def _odd_kernel_sizes_from_range(k_start: int, k_end: int, n_steps: int) -> List[int]:
    """
    Produce una lista di kernel sizes dispari tra k_start e k_end inclusi.
    Se n_steps == 1 ritorna [k_end].
    Garantisce che ogni valore sia un intero dispari >= 1.
    """
    k_start = int(max(1, round(k_start)))
    k_end   = int(max(1, round(k_end)))
    # forziamo kernel dispari
    if k_start % 2 == 0:
        k_start += 1
    if k_end % 2 == 0:
        k_end -= 1
    if k_end < k_start:
        k_end = k_start

    if n_steps <= 1:
        return [k_end]

    # generiamo n_steps valori equi-spaziati tra k_start e k_end
    arr = np.linspace(k_start, k_end, n_steps)
    ks = []
    for a in arr:
        k = int(round(a))
        if k % 2 == 0:
            k += 1
        if k < 1:
            k = 1
        ks.append(k)
    # rimuoviamo duplicati mantenendo ordine
    ks_unique = []
    for k in ks:
        if len(ks_unique) == 0 or ks_unique[-1] != k:
            ks_unique.append(k)
    return ks_unique


def apply_median_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Applica il filtro mediano con kernel quadrato di lato ksize.
    Se ksize <= 1 ritorna l'immagine originale.
    """
    if ksize <= 1:
        return img.copy()
    # OpenCV richiede ksize int dispari
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def attacks(input1: str, attack_name: Union[str, List[str]], param_array: dict):
    """
    Attacco median filter.

    - input1: path immagine (watermarked)
    - attack_name: 'median' oppure ['median']
    - param_array:
        {
          "k_start": int (default 3),
          "k_end": int (default 15),
          "n_steps": int (default 7),
          "out_dir": str (default "attacked_images_median")
        }

    Ritorna: lista di path generati (in ordine di increasing kernel size).
    """
    if isinstance(attack_name, list):
        names = attack_name
    else:
        names = [attack_name]

    if 'median' not in [n.lower() for n in names]:
        raise ValueError("Questo script supporta solo 'median' come attack_name.")

    k_start = int(param_array.get("k_start", 3))
    k_end   = int(param_array.get("k_end", 15))
    n_steps = int(param_array.get("n_steps", 7))
    out_dir = param_array.get("out_dir", "attacked_images_median")

    os.makedirs(out_dir, exist_ok=True)

    # carica immagine in grayscale
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {input1}")

    kernel_sizes = _odd_kernel_sizes_from_range(k_start, k_end, n_steps)
    out_paths = []
    base_name = os.path.splitext(os.path.basename(input1))[0]

    for k in kernel_sizes:
        filtered = apply_median_filter(img, k)
        fname = f"{base_name}_median_k{k}.png"
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, filtered)
        out_paths.append(out_path)

    return out_paths
