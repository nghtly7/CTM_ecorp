import os
import json
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


def attack_sharpen(img: np.ndarray, amount: float = 0.5, radius: float = 1.5) -> np.ndarray:
    """
    Unsharp mask attack.
    sharpen = clip(img + amount * (img - gaussian_blur(img, radius)))
    - amount: sharpening strength (e.g., 0.3..1.5)
    - radius: Gaussian sigma (controls blur radius)
    """
    sigma = max(0.1, float(radius))
    # Derive kernel size from sigma (approx 3*sigma per side)
    ksize = int(2 * round(3 * sigma) + 1)
    ksize = max(3, ksize)
    blur = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    high = cv2.addWeighted(img.astype(np.float32), 1.0 + float(amount), blur.astype(np.float32), -float(amount), 0)
    out = np.clip(high, 0, 255).astype(np.uint8)
    return out


def test_detection():
    # Configuration
    input_dir = 'sample-images'
    wm_dir_existing = 'watermarked_images_ROC'  # reuse if exists
    wm_dir = 'watermarked_images_det'
    att_dir = 'attacked_images_det'
    ensure_dir(wm_dir)
    ensure_dir(att_dir)

    # Optional: fix random seed for reproducibility of AWGN
    np.random.seed(12345)

    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} directory not found!")
        return

    # Collect images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        print(f"No image files found in {input_dir}/")
        return

    # Detection threshold info (mirrors detector logic)
    default_tau = 0.85
    tau = float(os.environ.get('DET_TAU', default_tau)) if str(os.environ.get('DET_TAU', '')).strip() not in ('', None) else default_tau

    print(f"Found {len(image_files)} image(s) in {input_dir}/")
    print(f"Detection threshold tau = {tau:.3f} (override with env DET_TAU)")
    print(f"Attack success criterion: NOT DETECTED and WPSNR >= 35 dB")
    print('-' * 80)

    # Watermark file path used by embedding (kept for API compliance)
    watermark = 'mark.npy'

    # Define attack sweeps (increasing intensity)
    # Note: For JPEG and resize, lower quality/scale means stronger attack
    sweeps = {
        'none': {
            'levels': [{}],
            'apply': lambda im, p: attack_none(im),
            'label': lambda p: ''
        },
        'awgn': {
            'levels': [{ 'sigma': s } for s in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]],
            'apply': lambda im, p: attack_awgn(im, p['sigma']),
            'label': lambda p: f"sigma={p['sigma']:.1f}"
        },
        'jpeg': {
            'levels': [{ 'q': q } for q in [90, 70, 50, 30, 10]],
            'apply': lambda im, p: attack_jpeg(im, p['q']),
            'label': lambda p: f"q={p['q']}"
        },
        'median': {
            'levels': [{ 'ksize': k } for k in [3, 5, 7, 9]],
            'apply': lambda im, p: attack_median(im, p['ksize']),
            'label': lambda p: f"ksize={p['ksize']}"
        },
        'resize': {
            'levels': [{ 'scale': s } for s in [0.95, 0.90, 0.85, 0.75, 0.60]],
            'apply': lambda im, p: attack_resize(im, p['scale']),
            'label': lambda p: f"scale={p['scale']:.2f}"
        },
        'sharpen': {
            # Higher amount => stronger sharpening; keep radius modest to avoid haloing
            'levels': [{ 'amount': a, 'radius': 1.5 } for a in [0.3, 0.6, 0.9, 1.2, 1.5]],
            'apply': lambda im, p: attack_sharpen(im, p['amount'], p['radius']),
            'label': lambda p: f"amount={p['amount']:.1f},radius={p['radius']}"
        },
    }

    # Global stats
    total_cases = 0
    total_detected = 0
    total_success_attacks = 0
    per_attack_stats = { name: {'cases': 0, 'detected': 0, 'success': 0, 'wpsnr_sum': 0.0} for name in sweeps.keys() }

    # Results store for JSON
    results = {
        'meta': {
            'tau': tau,
            'input_dir': input_dir,
            'wm_dir': wm_dir,
            'att_dir': att_dir,
            'success_criterion': 'NOT DETECTED and WPSNR >= 35 dB',
        },
        'images': [],
    }

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

        print(f"Processing image {i}/{len(image_files)}: {image_file}")

        image_record = {
            'image': image_file,
            'watermarked_path': wm_path,
            'attacks': {}
        }

        for atk_name, cfg in sweeps.items():
            levels = cfg['levels']
            apply_fn = cfg['apply']
            label_fn = cfg['label']

            print(f"  Attack: {atk_name} ({len(levels)} intensity level(s))")
            image_record['attacks'][atk_name] = []

            # Per-image, per-attack stats
            img_cases = 0
            img_detected = 0
            img_success = 0

            for li, params in enumerate(levels, 1):
                # Apply attack at this intensity
                att_img = apply_fn(wm_img, params)

                # Save attacked image with parameterized name
                if atk_name == 'none':
                    att_filename = f"{base}__{atk_name}.bmp"
                else:
                    att_filename = f"{base}__{atk_name}_{label_fn(params).replace('=', '').replace('.', 'p').replace(',', '-')}.bmp"
                att_path = os.path.join(att_dir, att_filename)
                cv2.imwrite(att_path, att_img)

                # Run detection
                out1, out2 = detection(orig_path, wm_path, att_path)

                # Stats
                img_cases += 1
                total_cases += 1
                per_attack_stats[atk_name]['cases'] += 1
                per_attack_stats[atk_name]['wpsnr_sum'] += float(out2)

                if out1 == 1:
                    img_detected += 1
                    total_detected += 1
                    per_attack_stats[atk_name]['detected'] += 1
                    status = 'DETECTED'
                    success_flag = ''
                else:
                    status = 'NOT DETECTED'
                    # Attack success if watermark destroyed (out1=0) AND WPSNR >= 35 dB
                    if out2 >= 35.0:
                        img_success += 1
                        total_success_attacks += 1
                        per_attack_stats[atk_name]['success'] += 1
                        success_flag = ' -> SUCCESS (WPSNR≥35)'
                    else:
                        success_flag = ''

                # Persist this case
                image_record['attacks'][atk_name].append({
                    'level': li,
                    'params': {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()},
                    'attacked_path': att_path,
                    'decision': int(out1),
                    'wpsnr': float(out2),
                    'status': status,
                    'success': bool(out1 == 0 and out2 >= 35.0),
                })

                print(f"    - L{li:02d}/{len(levels):02d} params[{label_fn(params)}] -> decision={out1}, WPSNR={out2:.2f} dB [{status}]{success_flag}")

            # End of levels for this attack on this image
            det_rate = img_detected / max(1, img_cases)
            print(f"    Summary for attack '{atk_name}' on {image_file}: cases={img_cases}, detected={img_detected} ({det_rate*100:.1f}%), successful attacks={img_success}")
            print()

    # save per-image record
    results['images'].append(image_record)

    # spacer between images
    print('-' * 80)

    # Global summary
    print('GLOBAL SUMMARY')
    print('-' * 80)
    print(f"Total attacked cases: {total_cases}")
    print(f"Detections (watermark present): {total_detected} ({(total_detected/max(1,total_cases))*100:.1f}%)")
    print(f"Successful attacks (not detected AND WPSNR>=35): {total_success_attacks}")
    print()
    for atk_name, st in per_attack_stats.items():
        cases = st['cases']
        if cases == 0:
            continue
        det = st['detected']
        succ = st['success']
        avg_wpsnr = st['wpsnr_sum'] / cases
        print(f"Attack '{atk_name}': cases={cases}, detected={det} ({(det/cases)*100:.1f}%), successful={succ}, avg WPSNR={avg_wpsnr:.2f} dB")

    # Attach summaries to JSON
    results['summary'] = {
        'total_cases': total_cases,
        'total_detected': total_detected,
        'total_success_attacks': total_success_attacks,
        'per_attack': {
            name: {
                'cases': st['cases'],
                'detected': st['detected'],
                'success': st['success'],
                'avg_wpsnr': (st['wpsnr_sum'] / st['cases']) if st['cases'] > 0 else None,
            }
            for name, st in per_attack_stats.items()
        },
    }

    # Write JSON file
    ensure_dir(att_dir)
    json_path = os.path.join(att_dir, 'attack_results.json')
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results written to {json_path}")
    except Exception as e:
        print(f"Failed to write JSON results: {e}")

    print('Done.')


if __name__ == '__main__':
    test_detection()
