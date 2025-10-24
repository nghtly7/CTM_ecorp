import os
import cv2
import numpy as np
from typing import List, Union, Optional

def awgn_image(img: np.ndarray, std: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Additive White Gaussian Noise (blind).
    img: uint8 grayscale (o multicanale) numpy array.
    std: standard deviation of Gaussian noise in pixel intensity units.
    seed: optional int for reproducibility.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.normal(0.0, std, img.shape)
    else:
        noise = np.random.normal(0.0, std, img.shape)

    img_f = img.astype(np.float64)
    attacked = img_f + noise
    attacked = np.clip(attacked, 0, 255)
    return attacked.astype(np.uint8)


def attacks(input1: str, attack_name: Union[str, List[str]], param_array: List[Union[float,int]]):
    """
    Generic attacks() function (competition style). Currently supports:
      - 'awgn' : param_array = [std] or [std, seed]
    Returns attacked image (numpy uint8).
    """
    img = cv2.imread(input1, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {input1}")

    if isinstance(attack_name, list):
        current = img
        for a in attack_name:
            if a.lower() == 'awgn':
                if len(param_array) == 0:
                    raise ValueError("AWGN needs at least std in param_array")
                std = float(param_array[0])
                seed = int(param_array[1]) if len(param_array) > 1 else None
                current = awgn_image(current, std, seed)
            else:
                raise ValueError(f"Unsupported attack: {a}")
        return current
    else:
        a = attack_name.lower()
        if a == 'awgn':
            if len(param_array) == 0:
                raise ValueError("AWGN needs at least std in param_array")
            std = float(param_array[0])
            seed = int(param_array[1]) if len(param_array) > 1 else None
            return awgn_image(img, std, seed)
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")


def batch_attack_folder(input_folder: str,
                        output_folder: str,
                        attack_name: str,
                        param_array: List[Union[float,int]],
                        extensions: List[str] = ['.bmp', '.png', '.jpg', '.jpeg']):
    """
    Apply attack to all images in input_folder and save in output_folder.
    Filenames preserved, prefixed with 'attacked_'.
    Returns list of (input_path, output_path).
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    processed = []
    for fname in sorted(os.listdir(input_folder)):
        if not any(fname.lower().endswith(ext) for ext in extensions):
            continue
        input_path = os.path.join(input_folder, fname)
        attacked_img = attacks(input_path, attack_name, param_array)
        out_name = f"attacked_{fname}"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, attacked_img)
        processed.append((input_path, out_path))
    return processed
