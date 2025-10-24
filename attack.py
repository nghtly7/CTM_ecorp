import os
import cv2
import numpy as np
import uuid
from pyparsing import Union
from rpds import List
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from PIL import Image
from skimage.transform import rescale

# -------------------------
# Attack implementations
# -------------------------


def _awgn(img: np.ndarray, std: float, seed: int, mean: float = 0.0) -> np.ndarray:
    """Additive White Gaussian Noise (AWGN)."""
    np.random.seed(seed)
    attacked = img.astype(np.float32) + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return np.uint8(attacked)


def _blur_gauss(img: np.ndarray, sigma: list) -> np.ndarray:
    """Gaussian blur."""
    attacked = gaussian_filter(img, sigma)
    return attacked


def _blur_median(img: np.ndarray, kernel_size: list) -> np.ndarray:
    """Median filter (ensure odd kernel sizes)."""
    kernel_size = [int(k) if int(k) % 2 == 1 else int(k) + 1 for k in kernel_size]
    if len(kernel_size) == 1:
        kernel_size = kernel_size[0]
    attacked = medfilt(img, kernel_size)
    return attacked


def _sharpening(img: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    """Unsharp masking (sharpen)."""
    img_f = img.astype(np.float32)
    filter_blurred_f = gaussian_filter(img_f, sigma)
    attacked_f = img_f + alpha * (img_f - filter_blurred_f)
    attacked_f = np.clip(attacked_f, 0, 255)
    return np.uint8(attacked_f)


def _resizing(img: np.ndarray, scale: float) -> np.ndarray:
    """Downscale then upscale to simulate resizing artifacts."""
    x, y = img.shape
    attacked_f = rescale(img, scale, anti_aliasing=True, mode='reflect')
    attacked_f = rescale(attacked_f, 1.0/scale, anti_aliasing=True, mode='reflect')
    attacked_f = np.clip(attacked_f * 255.0, 0, 255)
    attacked = cv2.resize(attacked_f, (y, x), interpolation=cv2.INTER_LINEAR)
    return np.uint8(attacked)


def _jpeg_compression(img: np.ndarray, QF: int) -> np.ndarray:
    """JPEG compression via temporary file."""
    tmp_filename = f'tmp_{uuid.uuid4()}.jpg'
    img_pil = Image.fromarray(img, mode="L")
    img_pil.save(tmp_filename, "JPEG", quality=int(QF))
    attacked = np.asarray(Image.open(tmp_filename), dtype=np.uint8)
    os.remove(tmp_filename)
    return attacked


def _canny_edge(img: np.ndarray, th1: int = 30, th2: int = 60) -> np.ndarray:
    """Canny edge detection (helper)."""
    d = 2
    edgeresult = img.copy()
    edgeresult = cv2.GaussianBlur(edgeresult, (2*d + 1, 2*d + 1), -1)[d:-d, d:-d]
    edgeresult = edgeresult.astype(np.uint8)
    edges = cv2.Canny(edgeresult, th1, th2)
    return edges


def _gauss_edge(img: np.ndarray, sigma: list, edge_th: list) -> np.ndarray:
    """Apply Gaussian blur only on detected edges."""
    edges = _canny_edge(img, th1=edge_th[0], th2=edge_th[1])
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    mask = (edges > 0).astype(np.uint8)
    blurred_img = _blur_gauss(img, sigma)
    attacked = (img * (1 - mask)) + (blurred_img * mask)
    return np.uint8(attacked)


def _gauss_flat(img: np.ndarray, sigma: list, edge_th: list) -> np.ndarray:
    """Apply Gaussian blur only on flat (non-edge) regions."""
    edges = _canny_edge(img, th1=edge_th[0], th2=edge_th[1])
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    mask = (edges > 0).astype(np.uint8)
    blurred_img = _blur_gauss(img, sigma)
    attacked = (img * mask) + (blurred_img * (1 - mask))
    return np.uint8(attacked)


# Main attack function
def attack(input1: str, attack_name: Union[str, List[str]], param_array: List) -> np.ndarray:
    """
    Apply one or more attacks sequentially to the grayscale image at input1.
    attack_name may be a single string or list of strings; param_array matches it.
    """
    img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {input1}")

    attacked_img = img.copy()

    if isinstance(attack_name, str):
        attack_names = [attack_name]
        param_list = [param_array]
    else:
        attack_names = attack_name
        param_list = param_array

    for name, params in zip(attack_names, param_list):
        name = name.lower().strip()
        try:
            if name == 'awgn':
                attacked_img = _awgn(attacked_img, std=params[0], seed=int(params[1]))
            elif name == 'blur':
                params = [params] if isinstance(params, (int, float)) else params
                attacked_img = _blur_gauss(attacked_img, sigma=params)
            elif name == 'sharp':
                attacked_img = _sharpening(attacked_img, sigma=params[0], alpha=params[1])
            elif name == 'median':
                params = [params] if isinstance(params, (int, float)) else params
                attacked_img = _blur_median(attacked_img, kernel_size=params)
            elif name == 'resize':
                attacked_img = _resizing(attacked_img, scale=float(params[0]))
            elif name == 'jpeg':
                attacked_img = _jpeg_compression(attacked_img, QF=int(params[0]))
            elif name == 'gauss_edge':
                attacked_img = _gauss_edge(attacked_img, sigma=params[0], edge_th=params[1])
            elif name == 'gauss_flat':
                attacked_img = _gauss_flat(attacked_img, sigma=params[0], edge_th=params[1])
            else:
                print(f"Warning: Attack '{name}' not recognized and will be skipped.")
        except Exception as e:
            print(f"Error applying attack '{name}' with params {params}: {e}. Skipping.")
            pass

    return attacked_img
