import os
import sys
import uuid
import importlib
import cv2

from attack_brute_force import attacks, save_attack_to_log

OUR_GROUP_NAME = "Ecorp"
ADV_GROUP_NAME = "Group_A"
ORIGINAL_NAME = "0002.bmp"

# ==================================
#    CUSTOMIZE YOUR ATTACK HERE
# ==================================
# Possible attacks & exact parameter formats:
# - jpeg:       [QF]                      -> QF: int (e.g. [70])
# - awgn:       [std, seed]               -> std: float, seed: int (e.g. [12.0, 123])
# - blur:       [sigma_y, sigma_x]        -> sigmas floats (e.g. [0.6, 0.6])
# - median:     [ky, kx]                  -> kernel sizes odd ints (e.g. [3, 3])
# - sharp:      [sigma, alpha]            -> sigma float, alpha float (e.g. [1.0, 1.5])
# - resize:     [scale]                   -> scale float (<1 downscale) (e.g. [0.8])
# - gauss_edge: [[sigma_y, sigma_x],[t1,t2]] -> localized blur on edges; edge thresholds for Canny
# - gauss_flat: [[sigma_y, sigma_x],[t1,t2]] -> localized blur on flat regions
#
# Single attack example:
#   attack_names = "jpeg"
#   param_values = [70]
#
# Multi-attack example (JPEG then Blur):
#   attack_names = ["jpeg","blur"]
#   param_values = [[70], [0.6, 0.6]]
#
# Multi-attack example (AWGN then JPEG):
#   attack_names = ["awgn","jpeg"]
#   param_values = [[10.0, 123], [50]]

attack_names = "jpeg"
param_values = [70]

RESULTS_FOLDER = "attack_results"


def _sanitize_filename_part(s: str) -> str:
    """Keep only safe filename characters: alnum, dash and underscore; replace others with '_'."""
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s)


def _num_to_safe_str(x) -> str:
    """Convert numbers to filesystem-safe short strings: 0.6 -> 0p6, 10 -> 10."""
    s = str(x)
    s = s.replace(".", "p")   # replace decimal point with 'p'
    s = s.replace(" ", "")    # remove spaces
    return s


def build_output_name(our_group: str, adv_group: str, orig_name: str, attack_names, param_values) -> str:
    """
    Build filename like:
      <our>_<adv>_<origbase>__attack1_p1_p2__attack2_p1.bmp
    Examples:
      jpeg [70] -> jpeg_70
      jpeg,blur [[70],[0.6,0.6]] -> jpeg_70__blur_0p6_0p6
    """
    # Normalize to lists
    if isinstance(attack_names, (list, tuple)):
        names = list(attack_names)
        params_list = list(param_values)
    else:
        names = [attack_names]
        params_list = [param_values]

    attack_parts = []
    for name, params in zip(names, params_list):
        # params may be scalar or iterable
        if isinstance(params, (list, tuple)):
            pstrs = [_num_to_safe_str(p) for p in params]
        else:
            pstrs = [_num_to_safe_str(params)]
        part = f"{name}_" + "_".join(pstrs)
        attack_parts.append(part)

    combined = "__".join(attack_parts)
    # sanitize final combined part and original base
    combined_safe = _sanitize_filename_part(combined)
    orig_base = _sanitize_filename_part(os.path.splitext(orig_name)[0])

    return f"{our_group}_{adv_group}_{orig_base}__{combined_safe}.bmp"


def main():
    # Ensure results folder exists
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Import adversary detection function
    detection_module_path = f"groups.{ADV_GROUP_NAME}.detection_{ADV_GROUP_NAME}"
    mod = importlib.import_module(detection_module_path)
    adversary_detection_function = getattr(mod, "detection")

    # Build portable paths
    ORIGINAL_PATH = os.path.join("sample-images", ORIGINAL_NAME)
    WATERMARKED_PATH = os.path.join("groups", ADV_GROUP_NAME, f"{ADV_GROUP_NAME}_{ORIGINAL_NAME}")

    if not os.path.exists(ORIGINAL_PATH):
        print(f"Error: Original image '{ORIGINAL_PATH}' does not exist.")
        sys.exit(1)
    if not os.path.exists(WATERMARKED_PATH):
        print(f"Error: Watermarked image '{WATERMARKED_PATH}' does not exist.")
        sys.exit(1)

    # Apply attack
    attacked_img = attacks(WATERMARKED_PATH, attack_names, param_values)
    if attacked_img is None:
        print("Error: attacks(...) returned None")
        sys.exit(1)

    # Save temporary attacked image for detection
    tmp_name = f"tmp_att_{uuid.uuid4().hex[:8]}.bmp"
    tmp_path = os.path.join(RESULTS_FOLDER, tmp_name)
    cv2.imwrite(tmp_path, attacked_img)

    # Call detection(original, watermarked, attacked_path)
    try:
        out = adversary_detection_function(ORIGINAL_PATH, WATERMARKED_PATH, tmp_path)
    except Exception as e:
        # cleanup temp and exit
        os.remove(tmp_path)
        print(f"Error calling detection(): {e}")
        sys.exit(1)

    found, wpsnr_val = out
    os.remove(tmp_path)

    if found == 0 and wpsnr_val >= 35.0:
        output_name = build_output_name(OUR_GROUP_NAME, ADV_GROUP_NAME, ORIGINAL_NAME, attack_names, param_values)
        output_path = os.path.join(RESULTS_FOLDER, output_name)
        cv2.imwrite(output_path, attacked_img)
        print(f"SUCCESS: attack successful. Saved attacked image as: {os.path.abspath(output_path)}")
        print(f"Found={found}, WPSNR={wpsnr_val:.2f} dB")

        csv_result = os.path.join(RESULTS_FOLDER, "result.csv")
        attack = {
            "names": attack_names,
            "params": param_values
        }
        save_attack_to_log(ADV_GROUP_NAME, ORIGINAL_NAME, attack, wpsnr_val, csv_result)
    else:
        print(f"Attack not successful. Found={found}, WPSNR={wpsnr_val:.2f} dB")
        sys.exit(0)


if __name__ == "__main__":
    main()
