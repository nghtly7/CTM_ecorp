import os
import cv2
import numpy as np
from embedding.Ecorp_strategy import embedding
from wpsnr import wpsnr

# try to import the detection module we prepared earlier
try:
    from detection.detection_ecorp_3 import detection
except Exception as e:
    detection = None
    print("Warning: couldn't import detection_ecorp_3.detection(). Detection will be skipped.")
    print("Import error:", e)

def test_embedding_and_detection():
    """Test embedding on all images in images/ and run detection to check presence of watermark"""
    output_dir = "watermarked_images_ROC"
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

    watermark = "mark.npy"  # default watermark file (detection uses the same by default)

    for idx, image_file in enumerate(image_files, start=1):
        input_path = os.path.join(input_dir, image_file)
        output_filename = f"watermarked_{image_file}"
        output_path = os.path.join(output_dir, output_filename)

        print(f"[{idx}/{len(image_files)}] Processing: {image_file}")

        try:
            # Run embedding (your embedding returns a numpy uint8 image)
            watermarked_image = embedding(input_path, watermark)

            # Save watermarked image
            saved = cv2.imwrite(output_path, watermarked_image)
            if not saved:
                print("  ✗ Failed to save watermarked image to", output_path)
                continue

            # Load original image for local WPSNR calculation
            original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                print("  ✗ Could not read original image for WPSNR calculation.")
                continue

            # Local wPSNR (using your wpsnr function)
            try:
                local_wpsnr = wpsnr(original_image, watermarked_image)
            except Exception:
                # fallback to simple PSNR if wpsnr fails
                mse = np.mean((original_image.astype(np.float64) - watermarked_image.astype(np.float64)) ** 2)
                local_wpsnr = float('inf') if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))

            print(f"  ✓ Watermarked image saved: {output_path}")
            print(f"  ✓ Local wPSNR (original vs watermarked): {local_wpsnr:.4f} dB")

            # Run detection if available
            if detection is not None:
                try:
                    # We pass attacked_path == watermarked image (no attack) to check presence right after embedding
                    dec, det_wpsnr, bit_accuracy = detection(input_path, output_path, output_path, watermark_path=watermark)
                except TypeError:
                    # in case the detection signature is different, try without watermark_path param
                    dec, det_wpsnr, bit_accuracy = detection(input_path, output_path, output_path)
                except Exception as e:
                    print("  ✗ Detection failed with exception:", e)
                    dec, det_wpsnr, bit_accuracy = None, None, None

                if dec is None:
                    print("  ✗ Detection did not return a valid result.")
                else:
                    status = "PRESENT" if dec == 1 else "NOT PRESENT"
                    print(f"  ▶ Detection result: {status} (dec = {dec})")
                    if bit_accuracy is not None:
                        print(f"    • Bit accuracy: {bit_accuracy*100:.2f}%")
                    if det_wpsnr is not None:
                        print(f"    • Detection wPSNR (watermarked vs attacked): {det_wpsnr:.4f} dB")
            else:
                print("  ⚠ Detection skipped (module not available).")

            print()

        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {e}")
            print()

    print("Processing completed!")

if __name__ == "__main__":
    test_embedding_and_detection()
