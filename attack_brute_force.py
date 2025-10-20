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
from scipy.ndimage import gaussian_filter
from PIL import Image
from skimage.transform import rescale
from collections import OrderedDict, defaultdict

OUR_GROUP_NAME = "ecorp"
ADV_GROUP_NAME = "..." # set the target adversary group name here
RESULTS_FOLDER = "attack_results"

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
    # """Produce ordered single and paired attack combinations (least -> most aggressive)."""
    # attack_params_db = OrderedDict([
    #     ('jpeg', ('QF', [[q] for q in range(100, 9, -5)])),

    #     ('blur', ('sigma', [[round(s, 2), round(s, 2)] for s in np.arange(0.2, 3.2, 0.2)])),

    #     ('median', ('kernel_size', [[3, 3], [5, 5], [7, 7], [9, 9]])),

    #     ('awgn', ('std_seed', [[s, 123] for s in [2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]]+ [[s, 42] for s in [5.0, 10.0, 20.0, 40.0]])),

    #     ('resize', ('scale', [[s] for s in [0.98, 0.95, 0.9, 0.85, 0.75, 0.5, 0.3]])),

    #     ('sharp', ('sigma_alpha', [
    #         [0.3, 0.3], [0.3, 0.6], [0.5, 0.5], [0.5, 1.0],
    #         [1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.5, 1.5],
    #         [2.0, 1.0], [2.0, 2.0]
    #     ])),

    #     ('gauss_edge', ('sigma_edge', [
    #         [[[0.3, 0.3], [20, 40]]],
    #         [[[0.5, 0.5], [30, 60]]],
    #         [[[1.0, 1.0], [30, 60]]],
    #         [[[1.5, 1.5], [50, 100]]],
    #         [[[2.0, 2.0], [50, 100]]],
    #     ])),

    #     ('gauss_flat', ('sigma_edge', [
    #         [[[0.3, 0.3], [20, 40]]],
    #         [[[0.5, 0.5], [30, 60]]],
    #         [[[1.0, 1.0], [30, 60]]],
    #         [[[1.5, 1.5], [50, 100]]],
    #         [[[2.0, 2.0], [50, 100]]],
    #     ])),
    # ])
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
    More robust and verbose for debugging.
    Returns (is_successful, wpsnr_val, attack_combo).
    """
    attack_names = attack_combo.get("names")
    attack_params = attack_combo.get("params")

    tmp_path = os.path.join(
        os.path.dirname(watermarked_path),
        f"tmp_attack_{uuid.uuid4()}.bmp"
    )

    # Debug header
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
        is_successful = (found == 0 and wpsnr_val >= 35.0)

        return (is_successful, float(wpsnr_val), attack_combo)

    except Exception as e:
        # Print full traceback + contextual info (very useful)
        print(f"[Worker ERROR] attack={attack_names} params={attack_params} -> {e}")
        traceback.print_exc()
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

    if best_attack is None:
        print("\n--- Brute-Force Complete ---")
        print("No successful attack found that meets WPSNR >= 35 criteria.")
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
