import os
import cv2
import numpy as np
from strategies.strategy_ecorp_1 import embedding, detection
# from strategies.strategy_ecorp_2 import embedding, detection
from wpsnr import wpsnr
from similarity import similarity
from threashold import compute_thr


def awgn(img, std, seed):
    mean = 0.0   # some constant
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return np.uint8(attacked)


def main():
    # --- Setup Directories and Parameters ---
    watermarked_dir = "watermarked_images"
    attacked_dir = "attacked_images"
    if not os.path.exists(watermarked_dir):
        os.makedirs(watermarked_dir)
    if not os.path.exists(attacked_dir):
        os.makedirs(attacked_dir)

    watermark_filepath = "mark.npy"
    if not os.path.exists(watermark_filepath):
        print(f"Error: Watermark file '{watermark_filepath}' not found!")
        return
    watermark = np.load(watermark_filepath)

    # Get all image files from the images directory
    input_dir = "images"
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.bmp')]
    if not image_files:
        print(f"No .bmp files found in '{input_dir}/'")
        return

    print(f"Found {len(image_files)} image(s) and starting watermark. Starting tests...")
    print("-" * 60)

    # Process each image
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        watermarked_path = os.path.join(watermarked_dir, f"w_{image_file}")
        attacked_path = os.path.join(attacked_dir, f"a_{image_file}")

        print(f"Processing image {i}: {image_file}")

        try:
            # Apply embedding
            watermarked_image = embedding(image, watermark)
            attacked_image = awgn(watermarked_image, 25, 42)

            # Save watermarked image
            cv2.imwrite(watermarked_path, watermarked_image)
            cv2.imwrite(attacked_path, attacked_image)

            # Calculate WPSNR
            watermarked_wpsnr = wpsnr(image, watermarked_image)
            attacked_wpsnr = wpsnr(image, attacked_image)

            wm_extracted = detection(image, watermarked_image)
            att_extracted = detection(image, attacked_image)

            similarity_wm = similarity(watermark, wm_extracted)
            similarity_att = similarity(watermark, att_extracted)

            tau, _ = compute_thr(similarity_wm, len(watermark), watermark, 1000)

            print(f"  ✓ Watermarked image saved: {watermarked_path}")
            print(f"  ✓ Attacked image saved: {attacked_path}")
            print(f"  ✓ Watermarked WPSNR: {watermarked_wpsnr:.4f} dB")
            print(f"  ✓ Attacked WPSNR: {attacked_wpsnr:.4f} dB")
            print(f"  ✓ Watermarked Similarity: {similarity_wm:.4f}")
            print(f"  ✓ Attacked Similarity: {similarity_att:.4f}")
            print(f"  ✓ Threshold (tau): {tau:.4f}")
            if similarity_att > tau:
                print("  ✓ Watermark detected in attacked image.")
            else:
                print("  ✗ Watermark NOT detected in attacked image.")
            print()

        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {str(e)}")
            print()

    print("Processing completed!")


if __name__ == "__main__":
    main()
