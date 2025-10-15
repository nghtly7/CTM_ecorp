import os
import cv2
import numpy as np
from embedding.embedding_ecorp_2 import embedding
from wpsnr import wpsnr

def test_embedding():
    """Test the embedding function on all images in the images/ folder"""
    
    # Create output directory if it doesn't exist
    output_dir = "watermarked_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Input directory
    input_dir = "images"
    
    # Check if images directory exists
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} directory not found!")
        return
    
    # Get all image files from the images directory
    image_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append(file)
    
    if not image_files:
        print(f"No image files found in {input_dir}/")
        return
    
    print(f"Found {len(image_files)} image(s) in {input_dir}/")
    print("-" * 50)
    
    # Dummy watermark file (not used by our embedding function)
    watermark = "mark.npy"

    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, image_file)
        output_filename = f"watermarked_{image_file}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing image {i}: {image_file}")
        
        try:
            # Apply embedding
            watermarked_image = embedding(input_path, watermark)
            
            # Save watermarked image
            cv2.imwrite(output_path, watermarked_image)
            
            # Load original image for WPSNR calculation
            original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            # Calculate WPSNR
            wpsnr_value = wpsnr(original_image, watermarked_image)
            
            print(f"  ✓ Watermarked image saved: {output_path}")
            print(f"  ✓ WPSNR: {wpsnr_value:.4f} dB")
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {str(e)}")
            print()
    
    
    print("Processing completed!")

if __name__ == "__main__":
    test_embedding()