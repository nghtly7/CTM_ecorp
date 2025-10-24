import os
import cv2
import numpy as np

from embedding.Ecorp_strategy import embedding
from detection.Ecorp_detection import detection


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def attack_none(img: np.ndarray) -> np.ndarray:
    return img.copy()


def attack_awgn(img: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def attack_jpeg(img: np.ndarray, q: int = 50) -> np.ndarray:
    # Encode to JPEG and decode back
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if not ok:
        return img.copy()
    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return dec


def attack_median(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img, k)


def attack_resize(img: np.ndarray, scale: float = 0.75) -> np.ndarray:
    h, w = img.shape
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def test_detection():
    # Configuration
    input_dir = 'sample-images'
    wm_dir_existing = 'watermarked_images_ROC'  # reuse if exists
    wm_dir = 'watermarked_images_det'
    att_dir = 'attacked_images_det'
    ensure_dir(wm_dir)
    ensure_dir(att_dir)

    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} directory not found!")
        return

    # Collect images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        print(f"No image files found in {input_dir}/")
        return

    print(f"Found {len(image_files)} image(s) in {input_dir}/")
    print('-' * 50)

    # Watermark file path used by embedding (kept for API compliance)
    watermark = 'mark.npy'

    # Define attacks to apply
    attacks = [
        ('none', lambda im: attack_none(im)),
        ('awgn8', lambda im: attack_awgn(im, 8.0)),
        ('jpeg50', lambda im: attack_jpeg(im, 50)),
        ('median5', lambda im: attack_median(im, 5)),
        ('resize075', lambda im: attack_resize(im, 0.75)),
    ]

    total = 0
    detected = 0
    success_attacks = 0  # attacks considered successful per rules

    for i, image_file in enumerate(image_files, 1):
        orig_path = os.path.join(input_dir, image_file)
        base = os.path.splitext(image_file)[0]

        # Get or create watermarked image
        wm_path = os.path.join(wm_dir_existing, f"watermarked_{image_file}")
        if not os.path.exists(wm_path):
            # Create via embedding
            wm_img = embedding(orig_path, watermark)
            wm_path = os.path.join(wm_dir, f"watermarked_{image_file}")
            cv2.imwrite(wm_path, wm_img)

        wm_img = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
        if wm_img is None:
            print(f"  ✗ Cannot read watermarked image: {wm_path}")
            continue

        print(f"Processing image {i}: {image_file}")

        for atk_name, atk_func in attacks:
            total += 1
            att_img = atk_func(wm_img)
            att_path = os.path.join(att_dir, f"{base}__{atk_name}.bmp")
            cv2.imwrite(att_path, att_img)

            # Run detection
            out1, out2 = detection(orig_path, wm_path, att_path)

            # out1: 1 if watermark present, else 0; out2: WPSNR(wm, attacked)
            # Attack success if watermark destroyed (out1=0) AND WPSNR >= 35 dB
            if out1 == 1:
                detected += 1
                status = 'DETECTED'
            else:
                status = 'NOT DETECTED'
                if out2 >= 35.0:
                    success_attacks += 1

            print(f"  - Attack {atk_name:<9} -> decision={out1}, WPSNR={out2:.2f} dB [{status}]")

        print()

    print('-' * 50)
    print(f"Total attacked cases: {total}")
    print(f"Detections (watermark present): {detected}")
    print(f"Successful attacks (not detected AND WPSNR>=35): {success_attacks}")
    print('Done.')


if __name__ == '__main__':
    test_detection()
