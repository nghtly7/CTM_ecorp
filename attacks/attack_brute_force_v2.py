import importlib
import os
import sys
import cv2
import numpy as np
import random
import inspect
import csv
import uuid
import glob
from itertools import zip_longest, combinations
from math import sqrt
from typing import List, Union, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import convolve2d, medfilt
from scipy.ndimage import gaussian_filter, median_filter
from PIL import Image
from skimage.transform import rescale
from collections import OrderedDict, defaultdict

OUR_GROUP_NAME = "ecorp"
ADV_GROUP_NAME = "..." # set the target adversary group name here
RESULTS_FOLDER = "attack_results"
WPSNR_TRESHHOLD = 15.0  # minimum WPSNR for successful attack

# Ensure project root is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Optional WPSNR import
try:
    from wpsnr import wpsnr
except ImportError:
    wpsnr = None
    print("Warning: wpsnr module not found. WPSNR calculation will be skipped.")
    
# -------------------------
# Verbosity control
# -------------------------
VERBOSE = False   # set True to enable per-attack debug prints
    
#! da collocare
def gaussian_feather_mask(h, w, by, bx, block, sigma=12):
    """
    Alias public-friendly per _gaussian_feather_mask — mantiene compatibilità con
    chiamate che usano il nome senza underscore.
    """
    return _gaussian_feather_mask(h, w, by, bx, block, sigma=sigma)
    
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


# -------------------------
# Attack dispatcher
# -------------------------
def attacks(input1: str, attack_name: Union[str, List[str]], param_array: List) -> np.ndarray:
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


# -------------------------
# Brute-force attack generator
# -------------------------
def _generate_attack_list() -> List[Dict[str, Any]]:
    """Produce ordered single and paired attack combinations (least -> most aggressive)."""
    attack_params_db = OrderedDict([
        ('jpeg', ('QF', [[q] for q in range(100, 9, -5)])),

        ('blur', ('sigma', [[round(s, 2), round(s, 2)] for s in np.arange(0.2, 3.2, 0.2)])),

        ('median', ('kernel_size', [[3, 3], [5, 5], [7, 7], [9, 9]])),

        ('awgn', ('std_seed', [[s, 123] for s in [2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]]+ [[s, 42] for s in [5.0, 10.0, 20.0, 40.0]])),

        ('resize', ('scale', [[s] for s in [0.98, 0.95, 0.9, 0.85, 0.75, 0.5, 0.3]])),

        ('sharp', ('sigma_alpha', [
            [0.3, 0.3], [0.3, 0.6], [0.5, 0.5], [0.5, 1.0],
            [1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.5, 1.5],
            [2.0, 1.0], [2.0, 2.0]
        ])),

        ('gauss_edge', ('sigma_edge', [
            [[[0.3, 0.3], [20, 40]]],
            [[[0.5, 0.5], [30, 60]]],
            [[[1.0, 1.0], [30, 60]]],
            [[[1.5, 1.5], [50, 100]]],
            [[[2.0, 2.0], [50, 100]]],
        ])),

        ('gauss_flat', ('sigma_edge', [
            [[[0.3, 0.3], [20, 40]]],
            [[[0.5, 0.5], [30, 60]]],
            [[[1.0, 1.0], [30, 60]]],
            [[[1.5, 1.5], [50, 100]]],
            [[[2.0, 2.0], [50, 100]]],
        ])),
    ])
    """Produce ordered single and paired attack combinations (least -> most aggressive)."""
    attack_params_db = OrderedDict([
        ('jpeg', ('QF', [[q] for q in range(95, 29, -5)])),
        ('blur', ('sigma', [[round(s, 2), round(s, 2)] for s in np.arange(0.4, 1.6, 0.2)])),
        ('median', ('kernel_size', [[3, 3], [5, 5]])),
        ('awgn', ('std_seed', [[s, 123] for s in [5.0, 10.0, 15.0, 20.0]])),
        ('resize', ('scale', [[s] for s in [0.9, 0.75, 0.5]])),
        ('sharp', ('sigma_alpha', [[0.5, 0.5], [0.5, 1.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5]])),
        ('gauss_edge', ('sigma_edge', [[[0.5, 0.5], [30, 60]], [[1.0, 1.0], [30, 60]]])),
        ('gauss_flat', ('sigma_edge', [[[0.5, 0.5], [30, 60]], [[1.0, 1.0], [30, 60]]])),
    ])

    single_attacks = []
    for name, (_, params_list) in attack_params_db.items():
        for p in params_list:
            single_attacks.append({"names": [name], "params": [p]})
    print(f"Generated {len(single_attacks)} single attack combinations.")

    attack_queue = single_attacks.copy()
    n = len(single_attacks)
    for i in range(n):
        for j in range(i+1, n):
            a1 = single_attacks[i]
            a2 = single_attacks[j]
            if a1["names"][0] != a2["names"][0]:
                attack_queue.append({"names": a1["names"] + a2["names"], "params": a1["params"] + a2["params"]})

    print(f"Generated {len(attack_queue)-len(single_attacks)} paired attack combinations.")
    print(f"Total attacks to test: {len(attack_queue)}")
    return attack_queue


# -------------------------
# Evaluation helpers
# -------------------------
import traceback

def _evaluate_single_attack(
    original_path: str,
    watermarked_path: str,
    adv_detection_func: Callable,
    attack_combo: Dict[str, Any]
) -> tuple:
    """
    Apply attack combo, run adversary detection, and compute WPSNR.
    Returns (is_successful, wpsnr_val, attack_combo).
    """
    attack_names = attack_combo.get("names")
    attack_params = attack_combo.get("params")

    tmp_path = os.path.join(
        os.path.dirname(watermarked_path),
        f"tmp_attack_{uuid.uuid4()}.bmp"
    )

    # Debug header (print only if VERBOSE)
    if VERBOSE:
        print(f"[Debug] Worker {os.getpid()} - testing attack: {attack_names} params: {attack_params}")

    try:
        # Validate inputs
        if not attack_names:
            raise ValueError("attack_combo['names'] is empty or missing.")
        if attack_params is None:
            raise ValueError("attack_combo['params'] is None.")

        # Apply attacks
        attacked_img = attacks(watermarked_path, attack_names, attack_params)

        # Validate attacked image
        if attacked_img is None:
            raise ValueError("attacks(...) returned None (failed to produce an image).")
        if not isinstance(attacked_img, (np.ndarray,)):
            raise ValueError(f"attacks(...) returned unexpected type: {type(attacked_img)}")

        # Save temp attacked image
        wrote = cv2.imwrite(tmp_path, attacked_img)
        if not wrote:
            raise IOError(f"cv2.imwrite failed for tmp_path={tmp_path}")

        # Run detection
        detection_out = adv_detection_func(original_path, watermarked_path, tmp_path)

        # Validate detection output
        if detection_out is None:
            raise ValueError("adv_detection_func returned None")
        if isinstance(detection_out, dict):
            # if somehow returns dict, try to extract expected fields
            if 'found' in detection_out and 'wpsnr' in detection_out:
                found = detection_out['found']
                wpsnr_val = detection_out['wpsnr']
            else:
                raise ValueError(f"adv_detection_func returned dict without 'found'/'wpsnr': {detection_out}")
        else:
            # assume iterable (found, wpsnr)
            try:
                found, wpsnr_val = detection_out
            except Exception as e:
                raise ValueError(f"adv_detection_func returned unexpected value: {detection_out}") from e

        # Ensure numeric wpsnr
        try:
            wpsnr_val = float(wpsnr_val)
        except Exception:
            raise ValueError(f"WPSNR value not convertible to float: {wpsnr_val}")

        # Success criteria: watermark not found (0) AND wpsnr is high
        is_successful = (found == 0 and wpsnr_val >= WPSNR_TRESHHOLD)

        return (is_successful, float(wpsnr_val), attack_combo)

    except Exception as e:
        # Print short contextual error (full traceback only if VERBOSE)
        if VERBOSE:
            print(f"[Worker ERROR] attack={attack_names} params={attack_params} -> {e}")
            traceback.print_exc()
        else:
            # minimal error log (keeps console cleaner)
            print(f"[Worker ERROR] attack={attack_names} -> {e}")
        return (False, 0.0, attack_combo)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _run_attack_type_worker(
    original_path: str,
    watermarked_path: str,
    adv_detection_func: Callable,
    attack_type_id: tuple,
    attack_combos_for_type: List[Dict[str, Any]]
) -> tuple:
    """
    Sequentially test variations of a given attack type and return the first success.
    """
    for attack_combo in attack_combos_for_type:
        try:
            success, wpsnr_val, attack = _evaluate_single_attack(
                original_path,
                watermarked_path,
                adv_detection_func,
                attack_combo
            )

            if success:
                print(f"  [Worker {os.getpid()}] +++ SUCCESS for type {attack_type_id} "
                      f"with {attack['names']} {attack['params']}. WPSNR: {wpsnr_val:.2f}")
                return (wpsnr_val, attack)

        except Exception as e:
            print(f"Error in worker for type {attack_type_id}: {e}")

    return (0.0, None)


def find_best_attack(
    original_path: str,
    watermarked_path: str,
    adv_detection_func: Callable,
    max_workers: int = 4
):
    """Parallel search across attack types to find the best successful attack."""
    print(f"\n--- Starting Brute-Force Attack Engine (Parallel by Type) ---\n")
    attack_list = _generate_attack_list()

    best_attack = None
    best_wpsnr = 0.0

    grouped_attacks = defaultdict(list)
    for attack_combo in attack_list:
        attack_type_id = tuple(sorted(attack_combo["names"]))
        grouped_attacks[attack_type_id].append(attack_combo)

    num_types_to_test = len(grouped_attacks)
    print(f"Grouped attacks into {num_types_to_test} unique types to be tested in parallel.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_attack_type_worker,
                original_path,
                watermarked_path,
                adv_detection_func,
                attack_type_id,
                attack_combos_for_type
            ): attack_type_id
            for attack_type_id, attack_combos_for_type in grouped_attacks.items()
        }

        for i, future in enumerate(as_completed(futures)):
            attack_type_id = futures[future]
            try:
                wpsnr_for_type, attack_for_type = future.result()
                print(f"  [Main] Job {i+1}/{num_types_to_test} complete (Type: {attack_type_id})")
                if attack_for_type is not None and wpsnr_for_type > best_wpsnr:
                    best_wpsnr = wpsnr_for_type
                    best_attack = attack_for_type
                    print("  *** New Best Overall Attack Found! ***")
                    print(f"  Attack: {best_attack['names']}")
                    print(f"  Params: {best_attack['params']}")
                    print(f"  WPSNR: {best_wpsnr:.2f}")
            except Exception as e:
                print(f"Error processing result for type {attack_type_id}: {e}")

            # --- TRY Greedy top-K local attack if no global single/paired attack succeeded ---
    if best_attack is None:
        print("[Main] No successful global attack found — trying Greedy top-K local attack...")
        try:
            success, wpsnr_local, info = greedy_top_k_local_attack(
                original_path,
                watermarked_path,
                adv_detection_func,
                block=64,
                K=12,
                wpsnr_thresh=WPSNR_TRESHHOLD,
                max_trials_per_patch=2,
                logger=print
            )
            if success:
                # we found a local solution; save as a pseudo-attack description
                best_attack = {
                    "names": ["greedy_local"],
                    "params": [info["steps"]],  # store steps for log / reproducibility
                }
                best_wpsnr = float(wpsnr_local)
                print(f"  [Main] Greedy local attack SUCCESS with WPSNR={best_wpsnr:.2f}")
        except Exception as e:
            print(f"[Main] Greedy attack failed with error: {e}")
            
    if best_attack is None:
        print("\n--- Brute-Force Complete ---")
        print("No successful attack found that meets WPSNR >=", WPSNR_TRESHHOLD, " criteria.")
        return None, 0.0
    else:
        print("\n--- Brute-Force Complete ---")
        print("Best attack found:")
        print(f"  Attack: {best_attack['names']}")
        print(f"  Params: {best_attack['params']}")
        print(f"  WPSNR: {best_wpsnr:.2f}")
        return best_attack, best_wpsnr


# -------------------------
# Logging utilities
# -------------------------
# --- BEGIN: Greedy top-k local attack integration ---
import tempfile
import itertools

def _detector_wrapper_write_and_call(adv_detection_func, orig_path, wm_path, candidate_img):
    """
    Save candidate_img to temp file and call adv_detection_func(original, watermarked, attacked)
    Returns (found:int, wpsnr:float, tmp_path:str)
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".bmp")
    os.close(fd)
    try:
        cv2.imwrite(tmp_path, candidate_img)
        # adv_detection_func expects (original_path, watermarked_path, attacked_path)
        found, wpsnr = adv_detection_func(orig_path, wm_path, tmp_path)
        return int(found), float(wpsnr), tmp_path
    except Exception as e:
        # on error, cleanup and rethrow
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass
        raise

def block_scores(img: np.ndarray, block: int = 64):
    """Compute simple combined score per block: var + entropy + midband DCT energy."""
    h, w = img.shape
    results = []
    for by in range(0, h, block):
        for bx in range(0, w, block):
            patch = img[by:by+block, bx:bx+block]
            if patch.size == 0:
                continue
            var = float(np.std(patch))
            hist = np.bincount(patch.flatten(), minlength=256).astype(np.float32)
            prob = hist / (patch.size + 1e-12)
            prob = np.clip(prob, 1e-12, 1.0)
            ent = -float(np.sum(prob * np.log2(prob)))
            # midband DCT energy (approx)
            dct_energy = 0.0
            ph, pw = patch.shape
            if ph >= 8 and pw >= 8:
                for y in range(0, ph - 7, 8):
                    for x in range(0, pw - 7, 8):
                        b8 = patch[y:y+8, x:x+8].astype(np.float32) - 128.0
                        d = cv2.dct(b8)
                        mid = d[1:5, 1:5]
                        dct_energy += float(np.sum(np.abs(mid)))
            else:
                dct_energy = float(np.sum(np.abs(cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F))))
            results.append(((by, bx), var, ent, dct_energy))
    if not results:
        return []
    arr_var = np.array([r[1] for r in results], dtype=float)
    arr_ent = np.array([r[2] for r in results], dtype=float)
    arr_dct = np.array([r[3] for r in results], dtype=float)
    def _norm(x):
        rng = np.ptp(x)
        if rng < 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / (rng + 1e-12)
    score = 0.4 * _norm(arr_var) + 0.3 * _norm(arr_dct) + 0.3 * _norm(arr_ent)
    scored = [ (results[i][0], float(score[i])) for i in range(len(results)) ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def _gaussian_feather_mask(h, w, by, bx, block, sigma=12):
    mask = np.zeros((h,w), dtype=np.float32)
    y0, y1 = by, min(by+block, h)
    x0, x1 = bx, min(bx+block, w)
    mask[y0:y1, x0:x1] = 1.0
    mask = gaussian_filter(mask, sigma=sigma)
    if mask.max() > 0:
        mask = mask / mask.max()
    return mask

def apply_local_attack(img, by, bx, block, attack_type='awgn', params=None, blend_sigma=12):
    if params is None:
        params = {}

    h,w = img.shape
    attacked = img.astype(np.float32).copy()

    y0,y1 = by, min(by+block, h)
    x0,x1 = bx, min(bx+block, w)
    patch = img[y0:y1, x0:x1].astype(np.float32)

    # compute replacement patch (same shape as patch)
    if attack_type == 'awgn':
        sigma = float(params.get('sigma', 6.0))
        replacement_patch = patch + np.random.normal(0, sigma, patch.shape)
    elif attack_type == 'blur':
        sigma = float(params.get('sigma', 1.0))
        ksize = max(3, int(2*round(3*sigma)+1))
        replacement_patch = cv2.GaussianBlur(patch.astype(np.float32), (ksize,ksize), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    elif attack_type == 'sharpen':
        amount = float(params.get('amount', 1.0))
        sigma = float(params.get('sigma', 1.0))
        ksize = max(3, int(2*round(3*sigma)+1))
        blurred = cv2.GaussianBlur(patch.astype(np.float32), (ksize,ksize), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
        replacement_patch = patch + amount * (patch - blurred)
    elif attack_type == 'median':
        k = int(params.get('k', 3))
        if k % 2 == 0: k += 1
        replacement_patch = median_filter(patch, size=k, mode='reflect')
    elif attack_type == 'resize':
        scale = float(params.get('scale', 0.9))
        ph, pw = patch.shape
        new_h = max(1, int(round(ph * scale)))
        new_w = max(1, int(round(pw * scale)))
        small = cv2.resize(patch.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        up = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_LINEAR)
        replacement_patch = up
    elif attack_type == 'jpeg':
        qf = int(params.get('qf', 85))
        patch_u8 = np.uint8(np.clip(patch,0,255))
        result, encimg = cv2.imencode('.jpg', patch_u8, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
        if result:
            dec = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            replacement_patch = dec
        else:
            replacement_patch = patch
    else:
        replacement_patch = patch

    # Create a full-size replacement image and copy the replacement_patch into it
    full_replacement = np.zeros_like(attacked, dtype=np.float32)
    full_replacement[y0:y1, x0:x1] = replacement_patch

    # Build the full-size mask and blend
    mask = gaussian_feather_mask(h, w, by, bx, block, sigma=blend_sigma)
    # ensure mask has same dims (h,w)
    if mask.ndim == 2:
        blend_mask = mask
    else:
        blend_mask = mask.squeeze()
    attacked = attacked * (1.0 - blend_mask) + full_replacement * blend_mask
    attacked = np.clip(attacked, 0, 255).astype(np.uint8)
    return attacked

def _apply_local_attack_to_image(img, by, bx, block, attack_type, params, blend_sigma=8):
    h,w = img.shape
    attacked_img = img.astype(np.float32).copy()
    y0,y1 = by, min(by+block, h)
    x0,x1 = bx, min(bx+block, w)
    patch = img[y0:y1, x0:x1].astype(np.float32)

    # produce replacement_patch (same shape as patch)
    if attack_type == 'awgn':
        std, seed = params.get('std', 6.0), int(params.get('seed', 42))
        rng = np.random.RandomState(seed)
        replacement_patch = patch + rng.normal(0, std, patch.shape)
    elif attack_type == 'blur':
        sigma = params.get('sigma', 1.0)
        # if you have a primitive that expects uint8 or whole-image, adapt as needed.
        replacement_patch = _blur_gauss(patch, sigma)  # ensure _blur_gauss accepts patch
    elif attack_type == 'median':
        k = int(params.get('k', 3))
        replacement_patch = _blur_median(patch, [k, k])
    elif attack_type == 'resize':
        scale = float(params.get('scale', 0.95))
        # _resizing might expect uint8 full image; adapt to patch:
        replacement_patch = _resizing(patch.astype(np.uint8), scale).astype(np.float32)
    elif attack_type == 'jpeg':
        qf = int(params.get('qf', 90))
        replacement_patch = _jpeg_compression(patch.astype(np.uint8), QF=qf).astype(np.float32)
    elif attack_type == 'sharpen':
        sigma = float(params.get('sigma', 1.0)); alpha = float(params.get('alpha', 1.0))
        replacement_patch = _sharpening(patch.astype(np.uint8), sigma=sigma, alpha=alpha).astype(np.float32)
    else:
        replacement_patch = patch

    # place replacement_patch into full-size replacement image
    full_replacement = np.zeros_like(attacked_img, dtype=np.float32)
    full_replacement[y0:y1, x0:x1] = replacement_patch

    mask = _gaussian_feather_mask(h, w, by, bx, block, sigma=blend_sigma)
    attacked_img = attacked_img * (1.0 - mask) + (full_replacement * mask)
    attacked_img = np.clip(attacked_img, 0, 255).astype(np.uint8)
    return attacked_img


def greedy_top_k_local_attack(original_path: str, watermarked_path: str, adv_detection_func: Callable,
                              block: int = 64, K: int = 12, wpsnr_thresh: float = WPSNR_TRESHHOLD,
                              max_trials_per_patch: int = 3, logger=print):
    """
    Greedy top-K local attack wrapper.
    Returns (success_bool, wpsnr, best_info_dict) where best_info_dict contains 'img' and 'steps' (list).
    """
    wm_img = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
    if wm_img is None:
        raise FileNotFoundError(f"Watermarked not found: {watermarked_path}")
    orig_path = original_path
    scores = block_scores(wm_img, block=block)
    top_blocks = [pos for (pos, sc) in scores[:K]]

    # default attacks_order (list of (atype, params_candidates))
    attacks_order = [
        ('awgn', [{'std': s, 'seed': sd} for s, sd in [(4.0,123),(6.0,42),(8.0,99)]]),
        ('blur', [{'sigma':[s,s]} for s in [0.8,1.4,2.2]]),
        ('median', [{'k':k} for k in [3,5]]),
        ('resize', [{'scale':s} for s in [0.98,0.95,0.9]]),
        ('jpeg', [{'qf':q} for q in [90,85,80]]),
        ('sharpen', [{'sigma':1.0,'alpha':a} for a in [0.8,1.2]])
    ]

    current_img = wm_img.copy()
    steps = []

    # quick pre-check
    found0, w0, tmp = _detector_wrapper_write_and_call(adv_detection_func, orig_path, watermarked_path, current_img)
    if found0 == 0 and w0 >= wpsnr_thresh:
        return True, w0, {'img': current_img, 'steps': steps}

    # iterate top blocks
    for (by, bx) in top_blocks:
        best_local = None
        for atype, candidates in attacks_order:
            # limit trials per_patch
            for cand in candidates[:max_trials_per_patch]:
                candidate_img = _apply_local_attack_to_image(current_img, by, bx, block, atype, cand)
                try:
                    found_c, wpsnr_c, tmp_path = _detector_wrapper_write_and_call(adv_detection_func, orig_path, watermarked_path, candidate_img)
                except Exception as e:
                    if logger: logger(f"[greedy] detector call failed: {e}")
                    continue
                # cleanup tmp
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

                info = {'patch':(by,bx), 'atype':atype, 'params':cand, 'found':found_c, 'wpsnr':wpsnr_c}
                if logger: logger(f"[greedy] test patch={by,bx} atype={atype} params={cand} -> found={found_c} wpsnr={wpsnr_c:.2f}")
                if best_local is None:
                    best_local = (found_c, wpsnr_c, candidate_img, info)
                else:
                    b_found, b_wpsnr = best_local[0], best_local[1]
                    # prefer lower found, then higher wpsnr
                    if found_c < b_found or (found_c == b_found and wpsnr_c > b_wpsnr):
                        best_local = (found_c, wpsnr_c, candidate_img, info)

                # immediate accept if successful
                if best_local and best_local[0] == 0 and best_local[1] >= wpsnr_thresh:
                    break
            if best_local and best_local[0] == 0 and best_local[1] >= wpsnr_thresh:
                break

        # commit best_local
        if best_local:
            bfnd, bw, bimg, binfo = best_local
            # accept if it improves found OR keeps same found but improves wpsnr
            cur_found, cur_wpsnr, _ = _detector_wrapper_write_and_call(adv_detection_func, orig_path, watermarked_path, current_img)
            if bfnd < cur_found or (bfnd == cur_found and bw >= cur_wpsnr - 0.5):
                current_img = bimg
                steps.append(binfo)
                if logger: logger(f"[greedy] accepted patch {binfo}")
        # test global
        found_g, wpsnr_g, tmp_g = _detector_wrapper_write_and_call(adv_detection_func, orig_path, watermarked_path, current_img)
        try:
            if os.path.exists(tmp_g): os.remove(tmp_g)
        except Exception:
            pass
        if logger: logger(f"[greedy] after commit -> found={found_g} wpsnr={wpsnr_g:.2f}")
        if found_g == 0 and wpsnr_g >= wpsnr_thresh:
            return True, wpsnr_g, {'img': current_img, 'steps': steps}

    # no success
    found_f, wpsnr_f, tmpf = _detector_wrapper_write_and_call(adv_detection_func, orig_path, watermarked_path, current_img)
    try:
        if os.path.exists(tmpf): os.remove(tmpf)
    except Exception:
        pass
    return (found_f == 0 and wpsnr_f >= wpsnr_thresh), wpsnr_f, {'img': current_img, 'steps': steps}
# --- END: Greedy top-k local attack integration ---


def _format_params_for_log(names: Any, params: Any) -> str:
    """Format attack names and parameters for CSV logging."""
    names_list = [names] if isinstance(names, str) else list(names)
    params_list = list(params) if isinstance(params, (list, tuple)) else [params]

    def to_items(p):
        return p if isinstance(p, (list, tuple)) else [p]

    parts = [
        f"{n}({'_'.join(map(str, to_items(p)))})"
        for n, p in zip_longest(names_list, params_list, fillvalue=[])
    ]
    return " + ".join(parts)


def save_attack_to_log(adv_group: str, image_name: str, best_attack: dict, wpsnr: float, csv_result: str):
    """Append best attack result to a CSV file."""
    try:
        file_exists = os.path.isfile(csv_result)
        with open(csv_result, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Image",
                    "Group",
                    "WPSNR",
                    "Attack(s) with parameters"
                ])
            image_basename = os.path.splitext(image_name)[0]
            attack_str = _format_params_for_log(best_attack["names"], best_attack["params"])
            writer.writerow([
                image_basename,
                adv_group,
                f"{wpsnr:.2f}",
                attack_str
            ])
        print(f"Successfully saved best attack result to log: '{csv_result}'")
    except IOError as e:
        print(f"Error: Could not write to log file '{csv_result}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during logging: {e}")


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    detection_module_path = f"groups.{ADV_GROUP_NAME}.detection_{ADV_GROUP_NAME}"
    mod = importlib.import_module(detection_module_path)
    adversary_detection_function = getattr(mod, "detection")

    NUM_WORKERS = os.cpu_count() or 4
    print(f"Using {NUM_WORKERS} parallel workers for brute-force attack (one worker per attack *type*).")

    base_dir = _ROOT_DIR
    PROJECT_ROOT = os.path.join(base_dir, "CTM_ecorp")
    ORIGINALS_DIR = os.path.join(PROJECT_ROOT, "sample-images")
    ADVERSARY_DIR = os.path.join(PROJECT_ROOT, "groups", ADV_GROUP_NAME)

    print(f"Attacking group '{ADV_GROUP_NAME}' in folder: {ADVERSARY_DIR}")
    print(f"Using originals from: {ORIGINALS_DIR}")

    attack_prefix = f"{ADV_GROUP_NAME}_"
    search_pattern = os.path.join(ADVERSARY_DIR, f"{attack_prefix}*.bmp")
    watermarked_image_paths = glob.glob(search_pattern)

    if not watermarked_image_paths:
        print(f"Error: No watermarked images found at {search_pattern}")
        sys.exit(1)

    print(f"Found {len(watermarked_image_paths)} images to attack.")

    for watermarked_path in watermarked_image_paths:
        watermarked_filename = os.path.basename(watermarked_path)
        original_image_name = watermarked_filename[len(attack_prefix):]
        original_path = os.path.join(ORIGINALS_DIR, original_image_name)

        if not os.path.exists(original_path):
            print(f"\nSkipping {watermarked_filename}:")
            print(f"  Corresponding original not found at {original_path}.")
            continue

        print(f"\n==============================")
        print(f"Processing image: {original_image_name}")
        print(f"==============================")
        print(f"  Original: {original_path}")
        print(f"  Watermarked: {watermarked_path}")

        best_attack_found, final_wpsnr = find_best_attack(
            original_path,
            watermarked_path,
            adversary_detection_function,
            max_workers=NUM_WORKERS
        )

        if best_attack_found:
            print("\nGenerating final attacked image with best attack...")
            final_attacked_img = attacks(
                watermarked_path,
                best_attack_found["names"],
                best_attack_found["params"]
            )

            output_filename = f"{OUR_GROUP_NAME}_{ADV_GROUP_NAME}_{original_image_name}"
            output_path = os.path.join(RESULTS_FOLDER, output_filename)
            cv2.imwrite(output_path, final_attacked_img)
            print(f"Done. '{output_path}' is ready to be uploaded.")

            csv_result = os.path.join(RESULTS_FOLDER, "result.csv")
            save_attack_to_log(
                ADV_GROUP_NAME,
                original_image_name,
                best_attack_found,
                final_wpsnr,
                csv_result
            )
        else:
            print(f"No successful attack found for {original_image_name}.")

    print("\n\nAll images processed. Attack run complete.")
