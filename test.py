import cv2
import numpy as np
import os
from paper_try_2.paper_embedding_v1 import embedding
from paper_try_2.paper_detection_v1_1 import extraction, detection, wpsnr, similarity
import random

def awgn(img, std, seed=None):
    mean = 0.0
    if seed is not None:
        np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked.astype(np.uint8)

def blur(img, sigma):
    from scipy.ndimage.filters import gaussian_filter
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
    attack_name = ""
    if i == 1:
        attacked = awgn(img, 3., 123)
        attack_name = "AWGN (std=3)"
    elif i == 2:
        attacked = blur(img, 3)
        attack_name = "Blur (sigma=3)"
    elif i == 3:
        attacked = sharpening(img, 1, 1)
        attack_name = "Sharpening"
    elif i == 4:
        attacked = median(img, 3)
        attack_name = "Median Filter (3x3)"
    elif i == 5:
        attacked = resizing(img, 0.8)
        attack_name = "Resizing (0.8)"
    elif i == 6:
        attacked = jpeg_compression(img, 75)
        attack_name = "JPEG Compression (QF=75)"
    elif i == 7:
        attacked = img.copy()
        attack_name = "No Attack"
    return attacked, attack_name


def test_watermarking_system(images_dir='images/', watermark_path='ecorp.npy'):
    """
    Comprehensive test of the watermarking system
    
    Args:
        images_dir: Directory containing test images
        watermark_path: Path to the watermark file (ecorp.npy)
    """
    
    # Load original watermark
    if not os.path.exists(watermark_path):
        raise FileNotFoundError(f"Watermark file not found: {watermark_path}")
    
    original_watermark = np.load(watermark_path).astype(np.uint8)
    print(f"Loaded watermark from {watermark_path}")
    print(f"Watermark size: {len(original_watermark)} bits\n")
    
    # Get list of images
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.bmp', '.png', '.jpg'))]
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    
    print(f"Found {len(image_files)} images in {images_dir}\n")
    print("="*80)
    
    # Process each image
    for img_idx, img_file in enumerate(image_files[:5], 1):  # Test first 5 images
        img_path = os.path.join(images_dir, img_file)
        print(f"\n{'='*80}")
        print(f"IMAGE {img_idx}/{min(5, len(image_files))}: {img_file}")
        print(f"{'='*80}\n")
        
        # Read original image
        I_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if I_orig is None:
            print(f"Error: Could not read {img_path}")
            continue
        
        cv2.imwrite('temp_original.bmp', I_orig)
        
        # ============================================================
        # TEST 1: Embed watermark and check WPSNR
        # ============================================================
        print("TEST 1: Embedding ecorp.npy watermark")
        print("-" * 80)
        
        I_watermarked = embedding('temp_original.bmp', watermark_path)
        cv2.imwrite('temp_watermarked.bmp', I_watermarked)
        
        wpsnr_embedded = wpsnr(I_orig.astype(np.float32) / 255.0, I_watermarked.astype(np.float32) / 255.0)
        print(f"✓ Watermark embedded")
        print(f"  WPSNR (original vs watermarked): {wpsnr_embedded:.2f} dB\n")
        
        # ============================================================
        # TEST 2: Extract watermark and check similarity
        # ============================================================
        print("TEST 2: Extracting watermark from watermarked image")
        print("-" * 80)
        
        extracted_watermark = extraction('temp_original.bmp', 'temp_watermarked.bmp')
        sim_extracted = similarity(original_watermark, extracted_watermark)
        hamming_dist = np.sum(np.abs(original_watermark - extracted_watermark))
        
        print(f"✓ Watermark extracted")
        print(f"  Similarity: {sim_extracted:.6f}")
        print(f"  Hamming Distance: {hamming_dist}/1024 bits\n")
        
        # ============================================================
        # TEST 3: Detection on watermarked image (no attack)
        # ============================================================
        print("TEST 3: Detection on watermarked image (no attack)")
        print("-" * 80)
        
        detected, wpsnr_val = detection('temp_original.bmp', 'temp_watermarked.bmp', 'temp_watermarked.bmp')
        print(f"  Detection result: {'✓ DETECTED' if detected == 1 else '✗ NOT DETECTED'}")
        print(f"  WPSNR: {wpsnr_val:.2f} dB\n")
        
        # ============================================================
        # TEST 4: Apply random attack and detect
        # ============================================================
        print("TEST 4: Detection after random attack")
        print("-" * 80)
        
        I_attacked, attack_name = random_attack(I_watermarked)
        cv2.imwrite('temp_attacked.bmp', I_attacked)
        
        wpsnr_attacked = wpsnr(I_orig.astype(np.float32) / 255.0, I_attacked.astype(np.float32) / 255.0)
        print(f"  Attack applied: {attack_name}")
        print(f"  WPSNR (original vs attacked): {wpsnr_attacked:.2f} dB")
        
        detected_attack, wpsnr_val_attack = detection('temp_original.bmp', 'temp_watermarked.bmp', 'temp_attacked.bmp')
        print(f"  Detection result: {'✓ DETECTED' if detected_attack == 1 else '✗ NOT DETECTED'}")
        print(f"  WPSNR (watermarked vs attacked): {wpsnr_val_attack:.2f} dB\n")
        
        # ============================================================
        # TEST 5: Detection with original image as third input
        # ============================================================
        print("TEST 5: Detection with original image as third input")
        print("-" * 80)
        
        detected_orig, wpsnr_orig = detection('temp_original.bmp', 'temp_watermarked.bmp', 'temp_original.bmp')
        print(f"  Detection result: {'✓ DETECTED' if detected_orig == 1 else '✗ NOT DETECTED'}")
        print(f"  WPSNR (watermarked vs original): {wpsnr_orig:.2f} dB")
        print(f"  Note: Should NOT detect (original has no watermark)\n")
        
        # ============================================================
        # TEST 6: Embed random watermark and test detection
        # ============================================================
        print("TEST 6: Detection with image watermarked with RANDOM watermark")
        print("-" * 80)
        
        # Generate and embed random watermark
        random_watermark = np.random.randint(0, 2, size=1024, dtype=np.uint8)
        np.save('temp_random_watermark.npy', random_watermark)
        
        I_random_watermarked = embedding('temp_original.bmp', 'temp_random_watermark.npy')
        cv2.imwrite('temp_random_watermarked.bmp', I_random_watermarked)
        
        # Test detection (comparing ecorp watermark in I_watermarked vs random watermark in I_random_watermarked)
        detected_random, wpsnr_random = detection('temp_original.bmp', 'temp_watermarked.bmp', 'temp_random_watermarked.bmp')
        
        # Also check similarity manually
        extracted_from_random = extraction('temp_original.bmp', 'temp_random_watermarked.bmp')
        sim_random = similarity(original_watermark, extracted_from_random)
        
        print(f"  Random watermark embedded in separate image")
        print(f"  Similarity (ecorp vs random): {sim_random:.6f}")
        print(f"  Detection result: {'✓ DETECTED' if detected_random == 1 else '✗ NOT DETECTED'}")
        print(f"  Note: Should NOT detect (different watermark)")
        print(f"  WPSNR: {wpsnr_random:.2f} dB\n")
        
    # Clean up temporary files
    temp_files = [
        'temp_original.bmp', 
        'temp_watermarked.bmp', 
        'temp_attacked.bmp',
        'temp_random_watermarked.bmp',
        'temp_random_watermark.npy'
    ]
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("\n" + "="*80)
    print("✓ All tests completed!")
    print("="*80)


if __name__ == "__main__":
    # Run the comprehensive test
    test_watermarking_system(
        images_dir='images/',
        watermark_path='ecorp.npy'
    )