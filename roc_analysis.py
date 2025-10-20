import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import uuid
import glob
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from PIL import Image
from skimage.transform import rescale
from sklearn.metrics import roc_curve, auc
from hashlib import sha256

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WATERMARK_PATH = os.path.join(_ROOT_DIR, 'mark.npy')
SAMPLE_IMAGES_DIR = os.path.join(_ROOT_DIR, 'sample-images')
NUM_ITERATIONS = 500
TARGET_FPR = 0.1

try:
    if _ROOT_DIR not in sys.path:
        sys.path.insert(0, _ROOT_DIR)
    from embedding_ecorp import embedding
    from detection_ecorp import (
        _read_gray_512, _alpha_matrix, _logpolar_fft, _phase_correlation_shift, _build_positions_and_dcts
    )
    try:
        import pywt
    except ImportError:
        pywt = None
        print("Warning: PyWavelets (pywt) not found. DWT will fallback to LL2 only.")

except ImportError as e:
    print(f"Error importing functions: {e}")
    print("Please ensure embedding_ecorp.py and detection_ecorp.py are in the same directory or accessible.")
    sys.exit(1)
# ---

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

random.seed(42)

def awgn(img, std):
    mean = 0.0
    attacked = img.astype(np.float32) + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return np.uint8(attacked)

def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked

def sharpening(img, sigma, alpha):
    img_f = img.astype(np.float32)
    filter_blurred_f = gaussian_filter(img_f, sigma)
    attacked_f = img_f + alpha * (img_f - filter_blurred_f)
    attacked = np.clip(attacked_f, 0, 255)
    return np.uint8(attacked)

def median(img, kernel_size):
    kernel_size_list = [int(k) if int(k) % 2 == 1 else int(k) + 1 for k in np.array([kernel_size]).flatten()]
    ksize = kernel_size_list[0] if len(kernel_size_list) == 1 else tuple(kernel_size_list)
    attacked = medfilt(img, ksize)
    return attacked

def resizing(img, scale):
    x, y = img.shape
    attacked_f = rescale(img.astype(np.float32)/255.0, scale, anti_aliasing=True, mode='reflect')
    attacked_f = rescale(attacked_f, 1.0/scale, anti_aliasing=True, mode='reflect')
    attacked = cv2.resize(attacked_f, (y, x), interpolation=cv2.INTER_LINEAR)
    attacked = np.clip(attacked * 255.0, 0, 255)
    return np.uint8(attacked)

def jpeg_compression(img, QF):
    tmp_filename = f'tmp_roc_{uuid.uuid4()}.jpg'
    img_pil = Image.fromarray(img)
    try:
        img_pil.save(tmp_filename, "JPEG", quality=int(QF))
        attacked = np.asarray(Image.open(tmp_filename), dtype=np.uint8)
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
    return attacked

def random_attack(img):
    attack_choice = random.randint(1, 6)
    if attack_choice == 1:
        std_dev = random.uniform(1.0, 15.0)
        print(f"Applying AWGN: std={std_dev:.2f}")
        attacked = awgn(img, std_dev)
    elif attack_choice == 2:
        sigma = random.uniform(0.5, 2.0)
        print(f"Applying Blur: sigma={sigma:.2f}")
        attacked = blur(img, sigma=[sigma, sigma])
    elif attack_choice == 3:
        sigma = random.uniform(0.5, 2.0)
        alpha = random.uniform(0.5, 1.5)
        print(f"Applying Sharpening: sigma={sigma:.2f}, alpha={alpha:.2f}")
        attacked = sharpening(img, sigma, alpha)
    elif attack_choice == 4:
        kernel_s = random.choice([3, 5])
        print(f"Applying Median: kernel={kernel_s}x{kernel_s}")
        attacked = median(img, kernel_size=kernel_s)
    elif attack_choice == 5:
        scale = random.uniform(0.5, 0.95)
        print(f"Applying Resizing: scale={scale:.2f}")
        attacked = resizing(img, scale)
    elif attack_choice == 6:
        quality = random.randint(30, 90)
        print(f"Applying JPEG: QF={quality}")
        attacked = jpeg_compression(img, quality)
    else:
        attacked = img
    return np.uint8(attacked)

# --- Extraction Function (Adapted from detection_ecorp) ---
def extract_ecc_bits(orig_path: str, wm_ref_img: np.ndarray, current_img: np.ndarray) -> np.ndarray:
    """Extracts ECC-encoded bits based on reference and current image."""
    I_wm = wm_ref_img # Already loaded reference image
    I_curr = current_img # Aligned or original watermarked image

    LP_ref, M_lp = _logpolar_fft(I_wm, out_size=(360, 200))
    aligned = I_curr
    try:
        LP_curr, _ = _logpolar_fft(I_curr, out_size=LP_ref.shape)
        dx, dy, resp = _phase_correlation_shift(LP_ref.astype(np.float32), LP_curr.astype(np.float32))
        cols = LP_ref.shape[1]
        rot_deg = -dx * 360.0 / float(cols)
        scale = float(np.exp(dy / max(M_lp, 1e-6)))
        # Only align if significant difference detected (same threshold as detection)
        if not (abs(rot_deg) < 0.2 and abs(scale - 1.0) < 0.002):
            center = (I_curr.shape[1] / 2.0, I_curr.shape[0] / 2.0)
            Maff = cv2.getRotationMatrix2D(center, rot_deg, 1.0 / max(scale, 1e-6))
            aligned = cv2.warpAffine(I_curr, Maff, (I_curr.shape[1], I_curr.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    except Exception:
        aligned = I_curr # Fallback if alignment fails

    subbands_names = ["LL2"] if pywt is None else ["LH2", "HL2"]

    # Reconstruct embedding state (deterministic part from detection)
    all_positions_ref, _ = _build_positions_and_dcts(I_wm, subbands_names) # Use reference to get positions
    total_positions = len(all_positions_ref)

    orig_len = 1024
    k, n = 11, 15
    blocks = int(np.ceil(orig_len / k))
    L_ecc = blocks * n

    R = max(1, min(32, total_positions // L_ecc))
    usable = R * L_ecc

    base_name = os.path.basename(orig_path).encode("utf-8")
    seed = int.from_bytes(sha256(base_name).digest()[:8], "big", signed=False) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)

    perm_idx = rng.permutation(L_ecc)
    pos_idx = rng.permutation(total_positions)[:usable]
    used_positions = [all_positions_ref[i] for i in pos_idx]

    alpha_mat = _alpha_matrix(0.008)
    sign_seq = np.where(rng.rand(usable) > 0.5, 1.0, -1.0).astype(np.float32)

    # Actual extraction using the *aligned* or *current* image's DCTs
    _, dct_subs_current = _build_positions_and_dcts(aligned, subbands_names)
    sums = np.zeros(L_ecc, dtype=np.float64)
    for i in range(usable):
        sgn = float(sign_seq[i])
        si, by, bx, u, v = used_positions[i]
        si_use = int(si)
        if si_use < 0 or si_use >= len(dct_subs_current):
            si_use = 0
        dct_arr = dct_subs_current[si_use]
        y0 = by * 8 + u
        x0 = bx * 8 + v
        coeff = 0.0
        if 0 <= y0 < dct_arr.shape[0] and 0 <= x0 < dct_arr.shape[1]:
            coeff = float(dct_arr[y0, x0])
        a = float(alpha_mat[u, v]) or 1.0
        sums[i // R] += (coeff / a) * sgn

    bits_perm = (sums > 0).astype(np.uint8)
    bits_ecc = np.zeros_like(bits_perm)
    bits_ecc[perm_idx] = bits_perm
    return bits_ecc # Return the ECC bits before decoding

# --- Main Execution Logic ---
if __name__ == "__main__":
    print("Starting ROC data generation...")
    scores = []
    labels = []

    original_image_paths = glob.glob(os.path.join(SAMPLE_IMAGES_DIR, '*.bmp')) + \
                           glob.glob(os.path.join(SAMPLE_IMAGES_DIR, '*.png'))
    if not original_image_paths:
        raise FileNotFoundError(f"No image files found in {SAMPLE_IMAGES_DIR}")
    print(f"Found {len(original_image_paths)} original images.")

    original_images = {}
    for path in original_image_paths:
        img = _read_gray_512(path) # Use your reader
        if img is None:
            print(f"Warning: Could not load image {path}. Skipping.")
        else:
            original_images[path] = img

    if not original_images:
         raise FileNotFoundError("Could not load any original images.")

    try:
        # We don't need W_original directly here, only its size for W_random
        W_original_payload = np.load(WATERMARK_PATH)
        mark_size = W_original_payload.size # Should be 1024
        if mark_size != 1024:
            print(f"Warning: Loaded watermark '{WATERMARK_PATH}' has size {mark_size}, expected 1024.")
        print(f"Loaded original watermark payload '{WATERMARK_PATH}' with size {mark_size}")
    except Exception as e:
        raise FileNotFoundError(f"Could not load watermark payload from {WATERMARK_PATH}: {e}")

    watermarked_images_arrays = {} # Store numpy arrays directly
    temp_wm_paths = {} # Store paths for extraction function
    print("Embedding watermark into original images...")
    for path, img in original_images.items():
        try:
            watermarked_img_array = embedding(path, WATERMARK_PATH)
            if watermarked_img_array is None or not isinstance(watermarked_img_array, np.ndarray):
                 raise ValueError(f"Embedding function did not return a valid NumPy array for {path}")

            temp_wm_path = f"temp_roc_wm_{os.path.basename(path)}_{uuid.uuid4()}.bmp"
            cv2.imwrite(temp_wm_path, watermarked_img_array)
            watermarked_images_arrays[path] = watermarked_img_array # Store array for attacks
            temp_wm_paths[path] = temp_wm_path # Store path for reference in extraction
        except Exception as e:
            print(f"ERROR during embedding for {path}: {e}")

    if not watermarked_images_arrays:
         raise RuntimeError("Embedding failed for all images. Cannot proceed.")
    print("Finished embedding.")

    # Determine ECC length L_ecc dynamically once (needed for W_random size)
    # Perform a dummy extraction on one watermarked image to get L_ecc
    try:
        dummy_orig_path = list(watermarked_images_arrays.keys())[0]
        dummy_wm_img = watermarked_images_arrays[dummy_orig_path]
        dummy_extracted_ecc = extract_ecc_bits(dummy_orig_path, dummy_wm_img, dummy_wm_img) # Extract from itself
        L_ecc = dummy_extracted_ecc.size
        print(f"Determined ECC block length (L_ecc): {L_ecc}")
    except Exception as e:
        print(f"Error determining L_ecc: {e}")
        L_ecc = 137 * 15 # Fallback based on calculation (ceil(1024/11)*15) = 94*15=1410
        print(f"Falling back to calculated L_ecc: {L_ecc}")
        # Need k, n for fallback calculation:
        k_ecc, n_ecc = 11, 15
        blocks_ecc = int(np.ceil(mark_size / k_ecc))
        L_ecc = blocks_ecc * n_ecc
        print(f"Falling back to calculated L_ecc: ceil({mark_size}/{k_ecc})*{n_ecc} = {L_ecc}")


    print(f"Starting {NUM_ITERATIONS} iterations for ROC score generation...")
    for i in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{NUM_ITERATIONS} ---")
        original_path = random.choice(list(watermarked_images_arrays.keys()))
        watermarked_img = watermarked_images_arrays[original_path]
        # watermarked_path = temp_wm_paths[original_path] # Not needed by extract_ecc_bits

        attacked_img = random_attack(watermarked_img.copy())
        # temp_att_path = f"temp_roc_att_{os.path.basename(original_path)}_{uuid.uuid4()}.bmp" # No need to save attacked image

        try:
            # Extract ECC bits from the original watermarked image (H1 reference)
            W_original_ecc = extract_ecc_bits(original_path, watermarked_img, watermarked_img)

            # Extract ECC bits from the attacked image
            W_extracted_ecc = extract_ecc_bits(original_path, watermarked_img, attacked_img)

            # Generate random ECC-length bit sequence for H0
            W_random_ecc = np.random.randint(0, 2, size=L_ecc)

            # Calculate similarity for H1 (Original ECC vs Extracted ECC)
            sim_h1 = similarity(W_original_ecc, W_extracted_ecc)
            scores.append(sim_h1)
            labels.append(1)
            print(f"H1 Similarity (Original ECC vs Extracted ECC): {sim_h1:.4f}")

            # Calculate similarity for H0 (Random ECC vs Extracted ECC)
            sim_h0 = similarity(W_random_ecc, W_extracted_ecc)
            scores.append(sim_h0)
            labels.append(0)
            print(f"H0 Similarity (Random ECC vs Extracted ECC):   {sim_h0:.4f}")

        except Exception as e:
            print(f"ERROR during extraction/similarity in iteration {i+1}: {e}")
        # No need to remove temp attacked file anymore

    print("\nFinished generating ROC scores.")

    for path in temp_wm_paths.values():
        if os.path.exists(path):
            os.remove(path)

    if not scores or not labels:
         print("ERROR: No scores were generated. Cannot compute ROC.")
    else:
        scores = np.asarray(scores)
        labels = np.asarray(labels)

        fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=True)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        valid_indices = np.where(fpr <= TARGET_FPR)[0]
        if len(valid_indices) == 0:
            print(f"Warning: No threshold found for FPR <= {TARGET_FPR}. Choosing the lowest FPR point.")
            idx = np.where(fpr > 0)[0]
            chosen_idx = idx[0] if len(idx) > 0 else 0
        else:
            chosen_idx = valid_indices[np.argmax(tpr[valid_indices])]

        chosen_threshold = thresholds[chosen_idx]
        chosen_fpr = fpr[chosen_idx]
        chosen_tpr = tpr[chosen_idx]

        print("\n--- Threshold Selection ---")
        print(f"Target maximum FPR: {TARGET_FPR}")
        print(f"Chosen Threshold (tau): {chosen_threshold:.4f}")
        print(f"Corresponding FPR:    {chosen_fpr:.4f}")
        print(f"Corresponding TPR:    {chosen_tpr:.4f}")

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.scatter(chosen_fpr, chosen_tpr, marker='o', color='red', s=100,
                    label=f'Chosen Point (FPR={chosen_fpr:.3f}, TPR={chosen_tpr:.3f})\nThreshold={chosen_threshold:.3f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve with Chosen Operating Point')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        print(f"\n===> REMEMBER TO USE THIS THRESHOLD VALUE IN YOUR DETECTION FUNCTION: {chosen_threshold:.4f} <===")