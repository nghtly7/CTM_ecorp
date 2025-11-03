# attacks/jpeg_attack.py
import os
import cv2
import numpy as np
from typing import List, Union

def _q_values(q_start: int, q_end: int, n_steps: int) -> List[int]:
    """
    Produce una lista di valori di qualità JPEG (interi 1..100).
    Se n_steps <= 1 ritorna [q_end].
    """
    q_start = int(np.clip(round(q_start), 1, 100))
    q_end   = int(np.clip(round(q_end), 1, 100))

    if n_steps <= 1:
        return [q_end]

    qs = np.linspace(q_start, q_end, n_steps)
    qs_int = [int(round(q)) for q in qs]
    # rimuoviamo duplicati mantenendo ordine
    qs_out = []
    for q in qs_int:
        if len(qs_out) == 0 or qs_out[-1] != q:
            qs_out.append(q)
    return qs_out

def attacks(input1: str, attack_name: Union[str, List[str]], param_array: dict):
    """
    Esegue una serie di compressioni JPEG sul file input1.

    - input1: path immagine watermarked (grayscale)
    - attack_name: 'jpeg' o ['jpeg']
    - param_array:
        {
          "q_start": int (1..100) default 95,
          "q_end":   int (1..100) default 30,
          "n_steps": int default 8,
          "out_dir": str default "attacked_images_jpeg",
          "force_gray": bool default True (salva in scala di grigi)
        }

    Ritorna: lista di path generati (ordina da q_start -> q_end)
    """
    if isinstance(attack_name, list):
        names = attack_name
    else:
        names = [attack_name]

    if 'jpeg' not in [n.lower() for n in names]:
        raise ValueError("Questo script supporta solo 'jpeg' come attack_name.")

    q_start = int(param_array.get("q_start", 95))
    q_end   = int(param_array.get("q_end", 30))
    n_steps = int(param_array.get("n_steps", 8))
    out_dir = param_array.get("out_dir", "attacked_images_jpeg")
    force_gray = bool(param_array.get("force_gray", True))

    os.makedirs(out_dir, exist_ok=True)

    # load image
    img = cv2.imread(input1, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {input1}")

    # If requested, convert to grayscale to avoid color subsampling differences
    if force_gray:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # keep image as-is; if it's grayscale it's fine, if BGR, we'll write as color JPEG
        pass

    qs = _q_values(q_start, q_end, n_steps)
    base_name = os.path.splitext(os.path.basename(input1))[0]
    out_paths = []

    for q in qs:
        fname = f"{base_name}_jpeg_q{int(q)}.jpg"
        out_path = os.path.join(out_dir, fname)

        # Use cv2.imencode to control quality and avoid surprising conversions
        # For grayscale images cv2.imencode will handle single-channel images correctly.
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
        success, encimg = cv2.imencode('.jpg', img, encode_param)
        if not success:
            # fallback to imwrite
            cv2.imwrite(out_path, img, encode_param)
        else:
            with open(out_path, 'wb') as f:
                f.write(encimg.tobytes())

        out_paths.append(out_path)

    return out_paths
