import cv2
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

from embedding_ecorp import embedding
from detection_ecorp import extraction
from similarity import similarity

# Basic attack implementations
def awgn(img, std, seed=None):
    mean = 0.0
    if seed is not None:
        np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked.astype(np.uint8)

def blur(img, sigma):
    from scipy.ndimage import gaussian_filter
    attacked = gaussian_filter(img, sigma)
    return attacked.astype(np.uint8)

def sharpening(img, sigma, alpha):
    from scipy.ndimage import gaussian_filter
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    attacked = np.clip(attacked, 0, 255)
    return attacked.astype(np.uint8)

def median(img, kernel_size):
    from scipy.signal import medfilt
    attacked = medfilt(img, kernel_size)
    return attacked.astype(np.uint8)

def resizing(img, scale):
    from skimage.transform import rescale
    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1/scale)
    attacked = attacked[:x, :y]
    attacked = np.clip(attacked * 255, 0, 255)
    return attacked.astype(np.uint8)

def jpeg_compression(img, QF):
    from PIL import Image
    img_pil = Image.fromarray(img)
    img_pil.save('tmp.jpg', "JPEG", quality=QF)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked, dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def random_attack(img):
    """Apply a random attack to the image"""
    i = random.randint(1, 7)
    if i == 1:
        attacked = awgn(img, 3., 123)
    elif i == 2:
        attacked = blur(img, 3)
    elif i == 3:
        attacked = sharpening(img, 1, 1)
    elif i == 4:
        attacked = median(img, 3)
    elif i == 5:
        attacked = resizing(img, 0.8)
    elif i == 6:
        attacked = jpeg_compression(img, 75)
    elif i == 7:
        attacked = img.copy()
    return attacked


def compute_roc_threshold(images_dir, num_iterations=500, max_fpr=0.1):
    """
    Compute ROC curve and determine optimal threshold for watermark detection
    
    Args:
        images_dir: Directory containing sample images
        num_iterations: Number of iterations for ROC computation
        max_fpr: Maximum acceptable False Positive Rate
        
    Returns:
        tau: Optimal threshold value
        fpr: False Positive Rates
        tpr: True Positive Rates
        roc_auc: Area Under Curve
    """
    
    # Get list of images
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.bmp'))]
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    
    print(f"Found {len(image_files)} images")
    
    # Arrays to store scores and labels for ROC
    scores = []
    labels = []
    
    # Set random seed for reproducibility
    np.random.seed(124)
    random.seed(3)
    
    # Load the original watermark from the correct location
    watermark_path = 'ecorp.npy'  # Adjust this path if needed
    if not os.path.exists(watermark_path):
        raise FileNotFoundError(f"Watermark file not found at {watermark_path}. Please ensure ecorp.npy exists.")
    
    original_watermark = np.load(watermark_path).astype(np.uint8)
    print(f"Loaded watermark from {watermark_path}")
    
    for iteration in range(num_iterations):
        # Select a random image
        img_file = random.choice(image_files)
        img_path = os.path.join(images_dir, img_file)
        
        # Read original image
        original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            print(f"Warning: Could not read {img_path}, skipping...")
            continue
            
        # Save original image temporarily
        cv2.imwrite('temp_original.bmp', original_image)
        
        # Embed watermark
        watermarked_image = embedding('temp_original.bmp', watermark_path)
        
        # Save watermarked image
        cv2.imwrite('temp_watermarked.bmp', watermarked_image)
        
        # Apply random attack
        attacked_image = random_attack(watermarked_image)
        cv2.imwrite('temp_attacked.bmp', attacked_image)
        
        # Extract watermark from attacked image
        extracted_watermark = extraction('temp_original.bmp', 'temp_attacked.bmp')
        
        # H1: Compute similarity between original and extracted watermark
        sim_h1 = similarity(original_watermark, extracted_watermark)
        scores.append(sim_h1)
        labels.append(1)
        
        # H0: Generate random watermark and compute similarity
        random_watermark = np.random.randint(0, 2, size=1024, dtype=np.uint8)
        sim_h0 = similarity(random_watermark, extracted_watermark)
        scores.append(sim_h0)
        labels.append(0)
        
        print(f"Iteration {iteration + 1}: sim(H1)={sim_h1:.4f}, sim(H0)={sim_h0:.4f}")

    scores_h1 = np.array(scores)[np.array(labels) == 1]
    scores_h0 = np.array(scores)[np.array(labels) == 0]

    #print min max e mean
    print(f"\nH1 Scores: min={scores_h1.min():.4f}, max={scores_h1.max():.4f}, mean={scores_h1.mean():.4f}")
    print(f"H0 Scores: min={scores_h0.min():.4f}, max={scores_h0.max():.4f}, mean={scores_h0.mean():.4f}")

    
    # Clean up temporary files
    for temp_file in ['temp_original.bmp', 'temp_watermarked.bmp', 'temp_attacked.bmp']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Convert to numpy arrays
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Compute ROC curve
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
    
    # Compute thresholds for additional FPR values
    target_fprs = [0.02, 0.01, 0]
    threshold_results = {}
    
    for target_fpr in target_fprs:
        valid_idx = np.where(fpr <= target_fpr)[0]
        if len(valid_idx) == 0:
            print(f"Warning: No threshold found with FPR <= {target_fpr}")
            threshold_results[target_fpr] = None
        else:
            idx = valid_idx[np.argmax(tpr[valid_idx])]
            threshold_results[target_fpr] = {
                'threshold': thresholds[idx],
                'fpr': fpr[idx],
                'tpr': tpr[idx]
            }
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', s=100, 
                label=f'Optimal (FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f})', zorder=5)
    
    # Plot additional thresholds
    colors = ['green', 'purple', 'orange']
    for i, (target_fpr, color) in enumerate(zip(target_fprs, colors)):
        if threshold_results[target_fpr] is not None:
            result = threshold_results[target_fpr]
            plt.scatter([result['fpr']], [result['tpr']], color=color, s=80,
                       label=f'FPR≤{target_fpr} (FPR={result["fpr"]:.3f}, TPR={result["tpr"]:.3f})', zorder=5)
    
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
    
    print("\n" + "-"*60)
    print("THRESHOLDS FOR DIFFERENT FPR VALUES")
    print("-"*60)
    for target_fpr in target_fprs:
        if threshold_results[target_fpr] is not None:
            result = threshold_results[target_fpr]
            print(f"\nFPR ≤ {target_fpr}:")
            print(f"  Threshold (tau): {result['threshold']:.6f}")
            print(f"  Actual FPR: {result['fpr']:.4f}")
            print(f"  TPR: {result['tpr']:.4f}")
        else:
            print(f"\nFPR ≤ {target_fpr}: No threshold found")
    print("="*60)
    
    # Save threshold to file
    np.save('optimal_threshold.npy', tau)
    print(f"\nThreshold saved to 'optimal_threshold.npy'")
    
    # Save all thresholds to a dictionary
    all_thresholds = {
        'optimal': {'threshold': tau, 'fpr': fpr[optimal_idx], 'tpr': tpr[optimal_idx], 'max_fpr': max_fpr}
    }
    all_thresholds.update({f'fpr_{fpr_val}': result for fpr_val, result in threshold_results.items() if result is not None})
    np.save('all_thresholds.npy', all_thresholds)
    print(f"All thresholds saved to 'all_thresholds.npy'")
    
    return tau, fpr, tpr, roc_auc, threshold_results


if __name__ == "__main__":
    # Compute ROC and get optimal threshold
    tau, fpr, tpr, roc_auc, additional_thresholds = compute_roc_threshold(
        images_dir='sample-images/',
        num_iterations=500,
        max_fpr=0.1
    )
    
    print("\n✓ ROC analysis complete!")
    #print(f"✓ Use tau = {tau:.6f} as your detection threshold")