import os
import cv2
import numpy as np
from embedding.Ecorp_strategy import embedding
from wpsnr import wpsnr

# import detection (deve restituire output1, output2)
try:
    from detection.detection_ecorp_3 import detection
except Exception as e:
    detection = None
    print("Warning: detection function not found. Skipping detection.")
    print("Import error:", e)

def test_embedding_and_detection():
    """Test embedding on all images in images/ and run detection to verify watermark presence."""
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
    print("-" * 60)

    watermark_file = "mark.npy"  # watermark used by embedding

    for idx, image_file in enumerate(image_files, start=1):
        input_path = os.path.join(input_dir, image_file)
        output_filename = f"watermarked_{image_file}"
        output_path = os.path.join(output_dir, output_filename)

        print(f"[{idx}/{len(image_files)}] Processing: {image_file}")

        try:
            # WATERMARKING
            watermarked_image = embedding(input_path, watermark_file)

            # SAVE WATERMARKED
            if not cv2.imwrite(output_path, watermarked_image):
                print("  ✗ Failed to save watermarked image:", output_path)
                continue

            # LOCAL WPSNR
            original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            local_wpsnr = wpsnr(original_image, watermarked_image)

            print(f"  ✓ Watermarked image saved: {output_path}")
            print(f"  ✓ Local wPSNR: {local_wpsnr:.4f} dB")

            # DETECTION
            if detection is not None:
                # at the moment attacked = watermarked → should ALWAYS detect = 1
                dec, det_wpsnr = detection(input_path, output_path, output_path)

                status = "PRESENT" if dec == 1 else "NOT PRESENT"
                print(f"  ▶ Detection result: {status} (dec = {dec})")
                print(f"    • Detection wPSNR: {det_wpsnr:.4f} dB")
            else:
                print("  ⚠ Detection skipped (module missing)")

            print()

        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {e}")
            print()

    print("Processing completed!")

if __name__ == "__main__":
    test_embedding_and_detection()
