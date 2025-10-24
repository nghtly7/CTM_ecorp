import traceback
import importlib
import os
import sys
import cv2
import numpy as np
import csv
import uuid
import glob
from itertools import zip_longest
from typing import List, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict, defaultdict

from attack import attack

OUR_GROUP_NAME = "ecorp"
ADV_GROUP_NAME = "v1"  # set the target adversary group name here
RESULTS_FOLDER = "attack_results"
ORIGINALS_DIR = "sample-images"

# Optional WPSNR import
try:
    from wpsnr import wpsnr
except ImportError:
    wpsnr = None
    print("Warning: wpsnr module not found. WPSNR calculation will be skipped.")

# -------------------------
# Brute-force attack generator
# -------------------------


def _generate_attack_list() -> List[Dict[str, Any]]:
    """Produce ordered single and paired attack combinations (least -> most aggressive)."""
    attack_params_db = OrderedDict([
        ('jpeg', ('QF', [[q] for q in range(50, 29, -5)])),
        ('blur', ('sigma', [[round(s, 2), round(s, 2)] for s in np.arange(0.4, 2.4, 0.4)])),
        ('median', ('kernel_size', [[3, 3], [5, 5]])),
        ('awgn', ('std_seed', [[s, 123] for s in [5.0, 10.0, 15.0, 20.0]])),
        ('resize', ('scale', [[s] for s in [0.7, 0.5]])),
        ('sharp', ('sigma_alpha', [[0.5, 0.5], [0.5, 1.0], [1.0, 0.5], [1.0, 1.0]])),
        ('gauss_edge', ('sigma_edge', [[[0.5, 0.5], [30, 60]], [[1.0, 1.0], [30, 60]]])),
        ('gauss_flat', ('sigma_edge', [[[0.5, 0.5], [30, 60]], [[1.0, 1.0], [30, 60]]])),
    ])
    # attack_params_db = OrderedDict([
    #     ('jpeg', ('QF', [[q] for q in range(95, 29, -5)])),
    #     ('blur', ('sigma', [[round(s, 2), round(s, 2)] for s in np.arange(0.4, 1.6, 0.2)])),
    #     ('median', ('kernel_size', [[3, 3], [5, 5]])),
    #     ('awgn', ('std_seed', [[s, 123] for s in [5.0, 10.0, 15.0, 20.0]])),
    #     ('resize', ('scale', [[s] for s in [0.9, 0.75, 0.5]])),
    #     ('sharp', ('sigma_alpha', [[0.5, 0.5], [0.5, 1.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5]])),
    #     ('gauss_edge', ('sigma_edge', [[[0.5, 0.5], [30, 60]], [[1.0, 1.0], [30, 60]]])),
    #     ('gauss_flat', ('sigma_edge', [[[0.5, 0.5], [30, 60]], [[1.0, 1.0], [30, 60]]])),
    # ])

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


def _evaluate_single_attack(
    original_path: str,
    watermarked_path: str,
    adv_detection_func: Callable,
    attack_combo: Dict[str, Any]
) -> tuple:
    """
    Apply attack combo, run adversary detection, and compute WPSNR.

    # Returns (status_code, wpsnr_val, attack_combo)
    # status_code:
    #   1 = SUCCESS (found=0, wpsnr >= 35)
    #   0 = FAIL_KEEP_GOING (found=1)
    #  -1 = FAIL_STOP_WORKER (found=0, wpsnr < 35)
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
        attacked_img = attack(watermarked_path, attack_names, attack_params)

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

        if wpsnr_val < 35.0:
            return (-1, wpsnr_val, attack_combo)  # -1 = FAIL_STOP_WORKER (too strong)

        # Return a status code instead of a simple boolean
        if found == 0:
            # Watermark removed AND quality is good
            return (1, wpsnr_val, attack_combo)  # 1 = SUCCESS

        # Watermark is still present
        return (0, wpsnr_val, attack_combo)   # 0 = FAIL_KEEP_GOING (not strong enough)

    except Exception as e:
        # Print full traceback + contextual info (very useful)
        print(f"[Worker ERROR] attack={attack_names} params={attack_params} -> {e}")
        traceback.print_exc()
        return (0, 0.0, attack_combo)
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
    Sequentially test variations of a given attack type.

    # Stops and returns the first SUCCESS (status=1)
    # Stops and breaks if it finds a FAIL_TOO_STRONG (status=-1)
    """
    print(f"   [Worker {os.getpid()}] Starting job for type {attack_type_id} ({len(attack_combos_for_type)} variations)")

    for attack_combo in attack_combos_for_type:
        try:
            status, wpsnr_val, attack = _evaluate_single_attack(
                original_path,
                watermarked_path,
                adv_detection_func,
                attack_combo
            )

            if status == 1:  # SUCCESS
                print(f"   [Worker {os.getpid()}] +++ SUCCESS for type {attack_type_id} "
                      f"with {attack['names']} {attack['params']}. WPSNR: {wpsnr_val:.2f}")
                return (wpsnr_val, attack)

            elif status == -1:  # FAIL_TOO_STRONG
                # This attack sequence is too strong and will only get stronger.
                # Stop processing this worker's queue.
                print(f"   [Worker {os.getpid()}] --- STOPPING type {attack_type_id} "
                      f"(Attack {attack['names']} {attack['params']} was too strong: WPSNR {wpsnr_val:.2f})")
                break  # Exit the for loop early

            if status == 0:  # FAIL_KEEP_GOING, the loop continues to the next attack_combo
                continue

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

        names = attack_combo["names"]
        params = attack_combo["params"]

        attack_type_id = None
        if len(names) == 1:
            # For single attacks, group by name
            # e.g., key = ('jpeg',)
            attack_type_id = (names[0],)
        else:
            # For paired attacks, group by:
            # 1. First attack name (e.g., 'jpeg')
            # 2. First attack param (e.g., '[95]')
            # 3. Second attack name (e.g., 'blur')
            # e.g., key = ('jpeg', '[95]', 'blur')

            param_str = str(params[0])
            attack_type_id = (names[0], param_str, names[1])

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

            if best_attack:
                attack_str = _format_params_for_log(best_attack["names"], best_attack["params"])
                wpsnr_str = f"{wpsnr:.2f}"
                log_message = f"Successfully saved best attack result to log: '{csv_result}'"
            else:
                attack_str = "NO_SUCCESSFUL_ATTACK_FOUND"
                wpsnr_str = "N/A"
                log_message = f"Successfully logged NO ATTACK for {image_basename} to log: '{csv_result}'"

            writer.writerow([
                image_basename,
                adv_group,
                wpsnr_str,
                attack_str
            ])
        print(log_message)
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

    ADVERSARY_DIR = os.path.join("groups", ADV_GROUP_NAME)
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
            final_attacked_img = attack(
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
