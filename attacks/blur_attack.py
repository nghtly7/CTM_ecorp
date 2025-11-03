import os
import cv2
import numpy as np
from typing import List, Union


def gaussian_kernel_size_from_sigma(sigma: float) -> int:
    """
    Converts sigma to a valid odd kernel size for GaussianBlur.
    """
    if sigma <= 0.0:
        return 1
    k = int(round(6.0 * float(sigma)))
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur with sigma. If sigma is too small, returns original.
    """
    ksize = gaussian_kernel_size_from_sigma(sigma)
    if ksize <= 1 or sigma <= 0.0:
        return img.copy()

    blurred = cv2.GaussianBlur(
        img,
        (ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    return blurred


def attacks(input1: str, attack_name: Union[str, List[str]], param_array: dict):
    """
    Perform BLUR attacks on a watermarked image.

    attack_name must be 'blur' or ['blur'].
    param_array must include:
        sigma_start, sigma_end, n_steps, out_dir
    """
    if isinstance(attack_name, list):
        names = attack_name
    else:
        names = [attack_name]

    if 'blur' not in [n.lower() for n in names]:
        raise ValueError("Questo script supporta solo 'blur'.")

    sigma_start = float(param_array.get("sigma_start", 0.5))
    sigma_end   = float(param_array.get("sigma_end", 8.0))
    n_steps     = int(param_array.get("n_steps", 6))
    out_dir     = param_array.get("out_dir", "attacked_images_blur")

    os.makedirs(out_dir, exist_ok=True)

    # load image
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {input1}")

    # build sigma range
    if n_steps <= 1:
        sigmas = np.array([sigma_end], dtype=float)
    else:
        sigmas = np.linspace(sigma_start, sigma_end, n_steps)

    base_name = os.path.splitext(os.path.basename(input1))[0]
    out_paths = []

    for s in sigmas:
        blurred = apply_gaussian_blur(img, float(s))
        fname = f"{base_name}_blur_sigma{float(s):.2f}.png"
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, blurred)
        out_paths.append(out_path)

    return out_paths
