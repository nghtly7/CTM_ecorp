import os
import cv2
import numpy as np
from paper_try_2.paper_embedding_v1 import embedding
from paper_try_2.paper_extraction import extraction
from wpsnr import wpsnr
from similarity import similarity

def test_embedding_and_extraction():
    """Test embedding on all images and extract watermark to verify."""
    output_dir = "watermarked_images"
    os.makedirs(output_dir, exist_ok=True)

    input_dir = "images"
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} directory not found!")
        return

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        print(f"No image files found in {input_dir}/")
        return

    print(f"Found {len(image_files)} image(s) in {input_dir}/")
    print("=" * 70)

    # Load original watermark
    watermark_file = "ecorp.npy"
    original_watermark = np.load(watermark_file).astype(np.float32)
    print(f"Original watermark loaded: {watermark_file} ({len(original_watermark)} bits)\n")

    for idx, image_file in enumerate(image_files, start=1):
        input_path = os.path.join(input_dir, image_file)
        output_filename = f"ecorp_{image_file}"
        output_path = os.path.join(output_dir, output_filename)

        print(f"[{idx}/{len(image_files)}] Processing: {image_file}")
        print("-" * 70)

        try:
            # 1) EMBEDDING
            watermarked_image = embedding(input_path, watermark_file)

            # 2) SAVE WATERMARKED IMAGE
            if not cv2.imwrite(output_path, watermarked_image):
                print("  ✗ Failed to save watermarked image:", output_path)
                continue
            print(f"  ✓ Watermarked image saved: {output_path}")

            # 3) CALCULATE WPSNR
            original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            wpsnr_value = wpsnr(original_image, watermarked_image)
            print(f"  ✓ WPSNR (original vs watermarked): {wpsnr_value:.4f} dB")

            # 4) EXTRACT WATERMARK (NON-BLIND: needs original image)
            extracted_watermark = extraction(output_path, input_path).astype(np.float32)
            print(f"  ✓ Watermark extracted: {len(extracted_watermark)} bits")

            # 5) CALCULATE SIMILARITY using provided function
            sim_value = similarity(original_watermark, extracted_watermark)
            print(f"  ✓ Similarity (extracted vs original): {sim_value:.4f}")

            # Additional info
            bit_errors = np.sum(extracted_watermark.astype(np.uint8) != original_watermark.astype(np.uint8))
            accuracy = (1024 - bit_errors) / 1024
            print(f"  → Bit accuracy: {accuracy*100:.2f}% ({bit_errors} errors)")
            
            print()

        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 70)
    print("Processing completed!")

if __name__ == "__main__":
    test_embedding_and_extraction()