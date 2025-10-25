import os
import cv2
import csv
import numpy as np

from paper_try_2.embedding_stronger import embedding
# from paper_try_2.embedding_stronger_v2 import embedding
# from jimaging_paper.embedding import embedding
# from paper_try_2.embedding_stronger import embedding

STRATEGY_NAME = "strongher_default"
# STRATEGY_NAME = "strongher_default_v2"
# STRATEGY_NAME = "jimaging_paper"
# STRATEGY_NAME = "AIIW_paper"

from wpsnr import wpsnr

def run_embedding():
    """Test the embedding function on all images in the sample-images/ folder
    and write a CSV with image name and WPSNR value.
    """
    # Create output directory if it doesn't exist
    output_dir = f"watermarked_images/{STRATEGY_NAME}"
    os.makedirs(output_dir, exist_ok=True)

    # CSV path
    csv_path = os.path.join(output_dir, "wpsnr_results.csv")
    # If CSV doesn't exist, write header
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["image_filename", "watermarked_filename", "wpsnr_dB"])

    # Input directory
    input_dir = "sample-images"

    # Check if images directory exists
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} directory not found!")
        csv_file.close()
        return

    # Get all image files from the images directory
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print(f"No image files found in {input_dir}/")
        csv_file.close()
        return

    print(f"Found {len(image_files)} image(s) in {input_dir}/")
    print("-" * 50)

    # Dummy watermark file (some embedding functions expect a path)
    watermark = "ecorp.npy"

    # Process each image
    for i, image_file in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, image_file)
        output_filename = f"{STRATEGY_NAME}_{image_file}"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Processing image {i}: {image_file}")

        try:
            # Apply embedding (embedding should return a 2D uint8 array)
            watermarked_image = embedding(input_path, watermark)

            # Validate returned image
            if watermarked_image is None:
                raise RuntimeError("embedding(...) returned None")
            if not isinstance(watermarked_image, np.ndarray):
                raise RuntimeError(f"embedding returned unexpected type: {type(watermarked_image)}")

            # Ensure uint8
            if watermarked_image.dtype != np.uint8:
                watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

            # Save watermarked image
            cv2.imwrite(output_path, watermarked_image)

            # Load original image for WPSNR calculation (convert to grayscale if needed)
            original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                raise FileNotFoundError(f"Original image could not be read: {input_path}")

            # Make sure shapes match; if embedding returned color convert or convert original accordingly
            if original_image.shape != watermarked_image.shape:
                # try converting watermarked to grayscale if it's color
                if watermarked_image.ndim == 3 and watermarked_image.shape[2] == 3:
                    watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
                else:
                    # resize watermarked to original (last resort)
                    watermarked_gray = cv2.resize(watermarked_image, (original_image.shape[1], original_image.shape[0]),
                                                 interpolation=cv2.INTER_LINEAR)
            else:
                # shapes ok
                watermarked_gray = watermarked_image

            # Calculate WPSNR (some implementations expect float arrays or uint8)
            try:
                wpsnr_value = float(wpsnr(original_image, watermarked_gray))
            except Exception:
                # fallback: try converting to float in [0,1]
                wpsnr_value = float(wpsnr(original_image.astype(np.float32)/255.0, watermarked_gray.astype(np.float32)/255.0))

            print(f"  ✓ Watermarked image saved: {output_path}")
            print(f"  ✓ WPSNR: {wpsnr_value:.4f} dB")
            print()

            # Write CSV row
            csv_writer.writerow([image_file, output_filename, f"{wpsnr_value:.4f}"])

        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {str(e)}")
            print()
            # still log error row with empty wpsnr
            csv_writer.writerow([image_file, "", "ERROR"])

    csv_file.close()
    print("Processing completed!")
    print(f"WPSNR results saved to: {csv_path}")


if __name__ == "__main__":
    run_embedding()
