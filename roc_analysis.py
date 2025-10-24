from itertools import product
from wpsnr import wpsnr
import pywt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from PIL import Image
from sklearn.metrics import roc_curve, auc
import random
import glob

from paper_try_2.paper_embedding_v1 import embedding
from paper_try_2.peper_detection import detection
from paper_try_2.paper_extraction import extraction

# --- SETTINGS ---
SAMPLE_IMAGES_DIR = 'sample-images'
WATERMARK_FILE = 'ecorp.npy'
NUM_ITERATIONS = 50
MARK_SIZE = 1024
random.seed(3)
np.random.seed(124)

# --- SIMILARITY FUNCTION ---


def similarity(X, X_star):
    return np.sum(X * X_star) / (np.sqrt(np.sum(X * X)) * np.sqrt(np.sum(X_star * X_star)))


# --- ATTACK FUNCTIONS ---
def awgn(img, std):
    attacked = img.astype(np.float32) + np.random.normal(0.0, std, img.shape)
    return np.clip(attacked, 0, 255).astype(np.uint8)


def blur(img, sigma):
    return np.clip(gaussian_filter(img, sigma), 0, 255).astype(np.uint8)


def sharpening(img, sigma, alpha):
    blurred = gaussian_filter(img, sigma)
    attacked = img.astype(np.float32) + alpha * (img.astype(np.float32) - blurred)
    return np.clip(attacked, 0, 255).astype(np.uint8)


def median(img, kernel_size):
    return medfilt(img, kernel_size).astype(np.uint8)


def resizing(img, scale):
    attacked = rescale(img.astype(np.float32)/255.0, scale, anti_aliasing=True, mode='reflect')
    attacked = rescale(attacked, 1.0/scale, anti_aliasing=True, mode='reflect')
    attacked = np.clip(attacked[:img.shape[0], :img.shape[1]] * 255, 0, 255)
    return attacked.astype(np.uint8)


def jpeg_compression(img, QF):
    tmp_filename = 'tmp.jpg'
    Image.fromarray(img).save(tmp_filename, "JPEG", quality=QF)
    attacked = np.array(Image.open(tmp_filename), dtype=np.uint8)
    os.remove(tmp_filename)
    return attacked


def _build_attack_param_grid():
    attack_params = {
        "awgn": [{"std": float(s)} for s in np.arange(10, 30, 5)],
        "blur": [{"sigma": float(s)} for s in np.arange(0.5, 4.5, 0.5)],
        "sharpen": [{"sigma": float(s), "alpha": float(a)} for s in [0.5, 1.0, 1.5] for a in [0.5, 1.0, 1.5]],
        "median": [{"kernel_size": k} for k in [3, 5, 7]],
        "resize": [{"scale": float(s)} for s in [0.9, 0.8, 0.7, 0.6]],
        "jpeg": [{"QF": int(q)} for q in [80, 75, 70, 60, 50, 40, 35, 30]],
    }
    return attack_params


def random_attack(orig_img, watermarked_img, min_wpsnr=35.0, max_wpsnr=53.0):
    attack_params_db = _build_attack_param_grid()
    attacks = list(attack_params_db.keys())
    random.shuffle(attacks)
    last_attacked = watermarked_img.copy()

    for attack in attacks:
        params_list = list(attack_params_db[attack])
        random.shuffle(params_list)  # shuffle parameters

        for params in params_list:
            # apply attack
            if attack == "awgn":
                attacked = awgn(watermarked_img, params["std"])
            elif attack == "blur":
                attacked = blur(watermarked_img, params["sigma"])
            elif attack == "sharpen":
                attacked = sharpening(watermarked_img, params["sigma"], params["alpha"])
            elif attack == "median":
                attacked = median(watermarked_img, params["kernel_size"])
            elif attack == "resize":
                attacked = resizing(watermarked_img, params["scale"])
            elif attack == "jpeg":
                attacked = jpeg_compression(watermarked_img, params["QF"])
            else:
                attacked = watermarked_img.copy()

            # compute WPSNR
            try:
                wpsnr_val = float(wpsnr(attacked, orig_img))
            except Exception:
                wpsnr_val = -1.0

            if min_wpsnr <= wpsnr_val <= max_wpsnr:
                print(f"Applied attack: {attack} with params {params}, WPSNR={wpsnr_val:.2f}")
                return attacked

            last_attacked = attacked

        if attack == "none":
            break
    # fallback if no attack satisfies WPSNR
    return last_attacked


def compute_roc_threshold(max_fpr=0.1):
    # --- LOAD DATA ---
    if not os.path.exists(WATERMARK_FILE):
        raise FileNotFoundError(f"Watermark file '{WATERMARK_FILE}' not found.")
    mark = np.load(WATERMARK_FILE)
    if mark.size != MARK_SIZE:
        raise ValueError(f"Watermark size {mark.size} != expected {MARK_SIZE}")

    image_paths = glob.glob(os.path.join(SAMPLE_IMAGES_DIR, '*.bmp')) + \
        glob.glob(os.path.join(SAMPLE_IMAGES_DIR, '*.png'))
    if not image_paths:
        raise FileNotFoundError(f"No images found in '{SAMPLE_IMAGES_DIR}'")

    # Get original images
    images = {}
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img.shape != (512, 512):
            img = cv2.resize(img, (512, 512))
        images[p] = img

    # --- ROC ANALYSIS ---
    scores = []
    labels = []

    # Embed watermark once per image
    watermarked_images = {}
    for path in image_paths:
        watermarked_images[path] = embedding(path, WATERMARK_FILE)
        print(f"Watermarked image: {path}")

    # Main ROC loop
    i = 0
    while (i < NUM_ITERATIONS):
        original_path = random.choice(list(images.keys()))
        original_img = images[original_path]
        watermarked_img = watermarked_images[original_path]

        attacked_img = random_attack(original_img, watermarked_img)

        wat_extracted = extraction(original_img, watermarked_img)
        wat_attacked = extraction(original_img, attacked_img)

        # Compute similarity with true watermark bits
        sim_extracted_att = similarity(wat_extracted, wat_attacked)
        sim_rand_att = similarity(np.random.randint(0, 2, MARK_SIZE), wat_attacked)

        print(f"Iteration {i}: sim(ex vs att)={sim_extracted_att:.4f}, sim(rand vs att)={sim_rand_att:.4f}")

        scores.append(sim_extracted_att)
        labels.append(1)
        scores.append(sim_rand_att)
        labels.append(0)
        i += 1

    # Summary statistics
    scores_real = np.array(scores)[np.array(labels) == 1]
    scores_rand = np.array(scores)[np.array(labels) == 0]
    print(
        f"Scores (ex vs att): min={scores_real.min():.4f}, max={scores_real.max():.4f}, mean={scores_real.mean():.4f}, std={scores_real.std():.4f}")
    print(
        f"Scores (rand vs att): min={scores_rand.min():.4f}, max={scores_rand.max():.4f}, mean={scores_rand.mean():.4f}, std={scores_rand.std():.4f}")

    # --- ROC PLOT ---
    scores = np.array(scores)
    labels = np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold for FPR <= max_fpr
    valid_indices = np.where(fpr <= max_fpr)[0]
    if len(valid_indices) == 0:
        print(f"Warning: No threshold found with FPR <= {max_fpr}")
        optimal_idx = 0
    else:
        # Among valid thresholds, choose the one with highest TPR
        optimal_idx = valid_indices[np.argmax(tpr[valid_indices])]

    tau = thresholds[optimal_idx]

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', s=100,
                label=f'Optimal (FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f})', zorder=5)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Watermark Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print results
    print("\n" + "="*60)
    print("ROC ANALYSIS RESULTS")
    print("="*60)
    print(f"Number of samples: {len(scores)}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"\nOptimal Threshold (tau): {tau:.6f}")
    print(f"Corresponding FPR: {fpr[optimal_idx]:.4f}")
    print(f"Corresponding TPR: {tpr[optimal_idx]:.4f}")
    print("="*60)

    return tau, fpr, tpr, roc_auc


if __name__ == "__main__":

    tau, fpr, tpr, roc_auc = compute_roc_threshold(
        max_fpr=0.1
    )

    print("\n✓ ROC analysis complete!")
    print(f"✓ Use tau = {tau:.6f} as your detection threshold")
