import importlib
import os
import sys
import cv2
import numpy as np
import random
import inspect
import csv
import uuid
from itertools import zip_longest, combinations
from math import sqrt
from typing import List, Union, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import convolve2d, medfilt
from scipy.ndimage import gaussian_filter
from PIL import Image
from skimage.transform import rescale


OUR_GROUP_NAME = "Ecorp"
ADV_GROUP_NAME = "Group_A"
ORIGINAL_NAME = "0002.bmp"
RESULTS_FOLDER = "attack_results"

# Ensure the root directory is in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from wpsnr import wpsnr
except ImportError:
    wpsnr = None
    print("Warning: wpsnr module not found. WPSNR calculation will be skipped.")


# =============================
#      LIST OF ATTACKS
# =============================

def _awgn(img: np.ndarray, std: float, seed: int, mean: float = 0.0) -> np.ndarray:
    """Adds Additive White Gaussian Noise (AWGN)."""
    np.random.seed(seed)
    attacked = img.astype(np.float32) + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return np.uint8(attacked)


def _blur_gauss(img: np.ndarray, sigma: list) -> np.ndarray:
    """Applies Gaussian blurring."""
    attacked = gaussian_filter(img, sigma)
    return attacked


def _blur_median(img: np.ndarray, kernel_size: list) -> np.ndarray:
    """Applies median filtering."""
    # Ensure kernel size is odd
    kernel_size = [int(k) if int(k) % 2 == 1 else int(k) + 1 for k in kernel_size]
    if len(kernel_size) == 1:
        kernel_size = kernel_size[0]
    attacked = medfilt(img, kernel_size)
    return attacked


def _sharpening(img: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    """Applies unsharp masking (sharpening)."""
    img_f = img.astype(np.float32)
    filter_blurred_f = gaussian_filter(img_f, sigma)
    attacked_f = img_f + alpha * (img_f - filter_blurred_f)
    attacked = np.clip(attacked_f, 0, 255)
    return np.uint8(attacked)


def _resizing(img: np.ndarray, scale: float) -> np.ndarray:
    """Applies resizing (downscaling and upscaling)."""
    x, y = img.shape
    attacked_f = rescale(img, scale, anti_aliasing=True, mode='reflect')
    attacked_f = rescale(attacked_f, 1.0/scale, anti_aliasing=True, mode='reflect')
    attacked_f = np.clip(attacked_f * 255.0, 0, 255)
    attacked = cv2.resize(attacked_f, (y, x), interpolation=cv2.INTER_LINEAR)
    return np.uint8(attacked)


def _jpeg_compression(img: np.ndarray, QF: int) -> np.ndarray:
    """Applies JPEG compression."""
    tmp_filename = f'tmp_{uuid.uuid4()}.jpg'
    img_pil = Image.fromarray(img, mode="L")
    img_pil.save(tmp_filename, "JPEG", quality=int(QF))
    attacked = np.asarray(Image.open(tmp_filename), dtype=np.uint8)
    os.remove(tmp_filename)
    return attacked


def _canny_edge(img: np.ndarray, th1: int = 30, th2: int = 60) -> np.ndarray:
    """Finds edges using Canny detector."""
    d = 2
    edgeresult = img.copy()
    edgeresult = cv2.GaussianBlur(edgeresult, (2*d + 1, 2*d + 1), -1)[d:-d, d:-d]
    edgeresult = edgeresult.astype(np.uint8)
    edges = cv2.Canny(edgeresult, th1, th2)
    return edges


def _gauss_edge(img: np.ndarray, sigma: list, edge_th: list) -> np.ndarray:
    """Applies Gaussian blur only to edge regions."""
    edges = _canny_edge(img, th1=edge_th[0], th2=edge_th[1])
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    mask = (edges > 0).astype(np.uint8)

    blurred_img = _blur_gauss(img, sigma)

    # Combine: blurred on edges, original on flat areas
    attacked = (img * (1 - mask)) + (blurred_img * mask)
    return np.uint8(attacked)


def _gauss_flat(img: np.ndarray, sigma: list, edge_th: list) -> np.ndarray:
    """Applies Gaussian blur only to flat regions."""
    edges = _canny_edge(img, th1=edge_th[0], th2=edge_th[1])
    edges = cv2.resize(edges, (img.shape[1], img.shape[0]))
    mask = (edges > 0).astype(np.uint8)

    blurred_img = _blur_gauss(img, sigma)

    # Combine: original on edges, blurred on flat areas
    attacked = (img * mask) + (blurred_img * (1 - mask))
    return np.uint8(attacked)


# =============================
#      ATTACK FUNCTION
# =============================

def attacks(input1: str, attack_name: Union[str, List[str]], param_array: List) -> np.ndarray:
    """
    input1: path to the watermarked image
    attack_name: single attack name (str) or list of attack names (List[str])
    param_array: parameters for the attack(s)
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

            # Localized attacks on edges or flat areas
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


# =============================
#   BRUTE-FORCE ATTACK ENGINE
# =============================

def _generate_attack_list() -> List[Dict[str, Any]]:
    attack_params_db = {
        'jpeg': ('QF',
                 [[q] for q in range(30, 81, 5)]),  # [30], [35], ..., [80]

        'blur': ('sigma',
                 [[s, s] for s in np.arange(0.4, 1.6, 0.2)]),  # [[0.4, 0.4]], ..., [[1.5, 1.5]]

        'median': ('kernel_size',
                   [[3, 3], [5, 5]]),

        'awgn': ('std_seed',
                 [[s, 123] for s in [5.0, 10.0, 15.0, 20.0]]),  # seed is fixed

        'resize': ('scale',
                   [[s] for s in [0.5, 0.75, 0.9]]),

        'sharp': ('sigma_alpha',
                  [[0.5, 0.5], [0.5, 1.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5]]),

        'gauss_edge': ('sigma_edge',
                       [[[0.5, 0.5], [30, 60]], [[1.0, 1.0], [30, 60]]]),

        'gauss_flat': ('sigma_edge',
                       [[[0.5, 0.5], [30, 60]], [[1.0, 1.0], [30, 60]]])
    }

    single_attacks = []
    for name, (param_key, param_values_list) in attack_params_db.items():
        for params_tuple in param_values_list:
            single_attacks.append(
                {"names": [name], "params": [params_tuple]}
            )

    print(f"Generated {len(single_attacks)} single attack combinations.")

    # Combine single attacks into pairs
    attack_queue = single_attacks.copy()

    for a1, a2 in combinations(single_attacks, 2):
        # Avoid repeating the same attack type in a pair
        if a1["names"][0] != a2["names"][0]:
            attack_queue.append({
                "names": a1["names"] + a2["names"],
                "params": a1["params"] + a2["params"]
            })

    print(f"Generated {len(attack_queue) - len(single_attacks)} paired attack combinations.")
    print(f"Total attacks to test: {len(attack_queue)}")
    return attack_queue


def _run_attack_worker(
    original_path: str,
    watermarked_path: str,
    adv_detection_func: Callable,
    attack_combo: Dict[str, Any]
) -> tuple:
    """
    Worker function to apply an attack combination and evaluate it
    """
    attack_names = attack_combo["names"]
    attack_params = attack_combo["params"]

    # Create a unique temp path for this worker
    tmp_path = os.path.join(
        os.path.dirname(original_path),
        f"tmp_attack_{uuid.uuid4()}.bmp"
    )

    try:
        attacked_img = attacks(watermarked_path, attack_names, attack_params)
        cv2.imwrite(tmp_path, attacked_img)
        found, wpsnr_val = adv_detection_func(original_path, watermarked_path, tmp_path)
        is_successful = (found == 0 and wpsnr_val >= 35.0)

        return (is_successful, float(wpsnr_val), attack_combo)

    except Exception as e:
        print(f"Worker Error with {attack_names}: {e}")
        return (False, 0.0, attack_combo)
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def find_best_attack(
    original_path: str,
    watermarked_path: str,
    adv_detection_func: Callable,
    max_workers: int = 4
):
    """
    Uses parallel processing to find the best attack combination that
    breaks the watermark (found=0) while maximizing WPSNR (>=35)
    """
    print("\n--- Starting Brute-Force Attack Engine ---\n")
    attack_list = _generate_attack_list()

    best_attack = None
    best_wpsnr = 0.0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(
                _run_attack_worker,
                original_path,
                watermarked_path,
                adv_detection_func,
                attack_combo
            ): attack_combo for attack_combo in attack_list
        }

        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            try:
                success, wpsnr_val, attack = future.result()

                print(f"  [Test {i+1}/{len(attack_list)}] "
                      f"Attack: {attack['names']} | "
                      f"Success: {success} | WPSNR: {wpsnr_val:.2f}")

                if success and wpsnr_val > best_wpsnr:
                    best_wpsnr = wpsnr_val
                    best_attack = attack
                    print(f"  *** New Best Attack Found! ***")
                    print(f"  Attack: {best_attack['names']}")
                    print(f"  Params: {best_attack['params']}")
                    print(f"  WPSNR: {best_wpsnr:.2f}")

            except Exception as e:
                print(f"Error processing result: {e}")

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


# =============================
#   LOGGING FUNCTIONALITY
# =============================
def _format_params_for_log(names: Any, params: Any) -> str:
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
    try:
        # Check if file exists to write header or not
        file_exists = os.path.isfile(csv_result)

        with open(csv_result, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header only if the file is new
            if not file_exists:
                writer.writerow([
                    "Image",
                    "Group",
                    "WPSNR",
                    "Attack(s) with parameters"
                ])

            # Format the data for the new row
            image_basename = os.path.splitext(image_name)[0]
            attack_str = _format_params_for_log(best_attack["names"], best_attack["params"])

            # Write the best attack data
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


if __name__ == "__main__":
    # Create results folder
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Import the adversary's detection function
    detection_module_path = f"groups.{ADV_GROUP_NAME}.detection_{ADV_GROUP_NAME}"
    mod = importlib.import_module(detection_module_path)
    adversary_detection_function = getattr(mod, "detection")

    # Input/output image paths
    ORIGINAL_PATH = os.path.join("sample-images", ORIGINAL_NAME)
    WATERMARKED_PATH = os.path.join("groups", ADV_GROUP_NAME, f"{ADV_GROUP_NAME}_{ORIGINAL_NAME}")

    if not os.path.exists(ORIGINAL_PATH):
        print(f"Error: Original image '{ORIGINAL_PATH}' does not exist.")
        sys.exit(1)
    if not os.path.exists(WATERMARKED_PATH):
        print(f"Error: Watermarked image '{WATERMARKED_PATH}' does not exist.")
        sys.exit(1)

    # Set number of parallel workers (CPU cores)
    NUM_WORKERS = os.cpu_count() or 4
    print(f"Using {NUM_WORKERS} parallel workers for brute-force attack.")

    # Run attacks
    best_attack_found, final_wpsnr = find_best_attack(
        ORIGINAL_PATH,
        WATERMARKED_PATH,
        adversary_detection_function,
        max_workers=NUM_WORKERS
    )

    # Generate final attacked image if best attack found
    if best_attack_found:
        print("\nGenerating final attacked image with best attack...")
        final_attacked_img = attacks(
            WATERMARKED_PATH,
            best_attack_found["names"],
            best_attack_found["params"]
        )
        output_path = os.path.join(RESULTS_FOLDER, f"{OUR_GROUP_NAME}_{ADV_GROUP_NAME}_{ORIGINAL_NAME}")
        cv2.imwrite(output_path, final_attacked_img)
        print(f"Done. '{output_path}' is ready to be uploaded.")

        csv_result = os.path.join(RESULTS_FOLDER, "result.csv")
        # Save best attack details to log
        save_attack_to_log(
            ADV_GROUP_NAME,
            ORIGINAL_NAME,
            best_attack_found,
            final_wpsnr,
            csv_result
        )
