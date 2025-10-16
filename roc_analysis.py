import os
import sys
import cv2
import numpy as np
from hashlib import sha256
from typing import Tuple

# Ensure project root for imports when run from CTM_ecorp/
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from detection.Ecorp_detection import (
    _read_gray_512,
    _alpha_matrix,
    _dwt2_level2,
    _zigzag_indices_8x8,
    _logpolar_fft,
    _phase_correlation_shift,
    _block_dct,
)

try:
    import pywt  # type: ignore
except Exception:
    pywt = None


def _build_positions(img_u8: np.ndarray, subbands_names=None):
    img_f32 = img_u8.astype(np.float32) / 255.0
    parts = _dwt2_level2(img_f32)
    if parts.get('_fallback', False):
        subs = [('LL2', parts['LL2'])]
    else:
        names = ['LL2'] if pywt is None else (subbands_names or ['LH2', 'HL2'])
        subs = [(nm, parts[nm]) for nm in names]

    zigzag = _zigzag_indices_8x8()
    sel = np.zeros((8, 8), dtype=bool)
    for k, (u, v) in enumerate(zigzag):
        if 10 <= k <= 30:
            sel[u, v] = True
    sel_positions = np.argwhere(sel)

    all_pos = []
    dct_subs = []
    for si, (_, sb) in enumerate(subs):
        H, W = sb.shape
        H8, W8 = H // 8 * 8, W // 8 * 8
        dct_arr = _block_dct(sb)
        dct_subs.append(dct_arr)
        by, bx = H8 // 8, W8 // 8
        for iy in range(by):
            for ix in range(bx):
                for (u, v) in sel_positions:
                    all_pos.append((si, iy, ix, int(u), int(v)))
    return all_pos, dct_subs


def detector_similarity(orig_path: str, wm_path: str, att_img: np.ndarray, tau: float = 0.82) -> Tuple[float, int]:
    I_wm = _read_gray_512(wm_path)
    I_att = att_img

    # Fourier–Mellin alignment (same as detector, but skip near-identity)
    LP_ref, M_lp = _logpolar_fft(I_wm, out_size=(360, 200))
    aligned = I_att
    try:
        LP_att, _ = _logpolar_fft(I_att, out_size=LP_ref.shape)
        dx, dy, resp = _phase_correlation_shift(LP_ref.astype(np.float32), LP_att.astype(np.float32))
        cols = LP_ref.shape[1]
        rot_deg = -dx * 360.0 / float(cols)
        scale = float(np.exp(dy / max(M_lp, 1e-6)))
        if not (abs(rot_deg) < 0.2 and abs(scale - 1.0) < 0.002):
            center = (I_att.shape[1] / 2.0, I_att.shape[0] / 2.0)
            Maff = cv2.getRotationMatrix2D(center, rot_deg, 1.0 / max(scale, 1e-6))
            aligned = cv2.warpAffine(I_att, Maff, (I_att.shape[1], I_att.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    except Exception:
        aligned = I_att

    subbands_names = ['LL2'] if pywt is None else ['LH2', 'HL2']
    all_positions_ref, _ = _build_positions(I_wm, subbands_names=subbands_names)
    total_positions = len(all_positions_ref)

    orig_len = 1024
    k, n = 11, 15
    blocks = int(np.ceil(orig_len / k))
    pad = blocks * k - orig_len
    L_ecc = blocks * n

    R = max(1, min(32, total_positions // L_ecc))
    usable = R * L_ecc

    base_name = os.path.basename(orig_path).encode('utf-8')
    seed = int.from_bytes(sha256(base_name).digest()[:8], 'big', signed=False) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)

    perm_idx = rng.permutation(L_ecc)
    pos_idx = rng.permutation(total_positions)[:usable]
    used_positions = [all_positions_ref[i] for i in pos_idx]

    alpha_mat = _alpha_matrix(0.008)
    sign_seq = np.where(rng.rand(usable) > 0.5, 1.0, -1.0).astype(np.float32)

    def extract(img_u8: np.ndarray):
        _, dct_subs = _build_positions(img_u8, subbands_names=subbands_names)
        sums = np.zeros(L_ecc, dtype=np.float64)
        for i in range(usable):
            sgn = float(sign_seq[i])
            si, by, bx, u, v = used_positions[i]
            si_use = int(si)
            if si_use < 0 or si_use >= len(dct_subs):
                si_use = 0
            dct_arr = dct_subs[si_use]
            y0 = by * 8 + u
            x0 = bx * 8 + v
            coeff = 0.0
            if 0 <= y0 < dct_arr.shape[0] and 0 <= x0 < dct_arr.shape[1]:
                coeff = float(dct_arr[y0, x0])
            a = float(alpha_mat[u, v]) or 1.0
            sums[i // R] += (coeff / a) * sgn
        bits_perm = (sums > 0).astype(np.uint8)
        bits_ecc = np.zeros_like(bits_perm)
        bits_ecc[perm_idx] = bits_perm
        return bits_ecc

    bits_ecc_ref = extract(I_wm)
    bits_ecc_att = extract(aligned)
    a = (bits_ecc_att.astype(np.float32) * 2.0 - 1.0)
    b = (bits_ecc_ref.astype(np.float32) * 2.0 - 1.0)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    dec = 1 if sim >= tau else 0
    return sim, dec


def _attack_awgn(img: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _attack_jpeg(img: np.ndarray, q: int = 50) -> np.ndarray:
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if not ok:
        return img.copy()
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def _attack_median(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = int(ksize) | 1
    return cv2.medianBlur(img, k)


def _attack_resize(img: np.ndarray, scale: float = 0.75) -> np.ndarray:
    h, w = img.shape
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def main():
    proj_root = _ROOT
    sample_dir = os.path.join(proj_root, 'sample-images')
    wm_dir = os.path.join(proj_root, 'watermarked_images_ROC')

    if not os.path.exists(sample_dir) or not os.path.exists(wm_dir):
        print('Missing sample-images/ or watermarked_images_ROC/ in', proj_root)
        sys.exit(1)

    attacks = {
        'none': lambda im: im.copy(),
        'awgn8': lambda im: _attack_awgn(im, 8.0),
        'jpeg50': lambda im: _attack_jpeg(im, 50),
        'median5': lambda im: _attack_median(im, 5),
        'resize075': lambda im: _attack_resize(im, 0.75),
    }

    image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    scores = []
    for i, name in enumerate(image_files, 1):
        orig_path = os.path.join(sample_dir, name)
        wm_path = os.path.join(wm_dir, f"watermarked_{name}")
        if not os.path.exists(wm_path):
            # Skip if pre-watermarked not present
            continue
        wm_img = _read_gray_512(wm_path)
        for atk_name, atk_func in attacks.items():
            att_img = atk_func(wm_img)
            sim, dec = detector_similarity(orig_path, wm_path, att_img, tau=0.82)
            scores.append((name, atk_name, sim, int(dec)))

    if not scores:
        print('No scores computed. Ensure watermarked_images_ROC is populated.')
        sys.exit(1)

    pos_attacks = {'none', 'awgn8', 'jpeg50', 'resize075'}
    neg_attacks = {'median5'}
    pos_sims = [s for (_, atk, s, _) in scores if atk in pos_attacks]
    neg_sims = [s for (_, atk, s, _) in scores if atk in neg_attacks]

    ths = np.linspace(0.6, 0.99, 100)
    roc = []
    for t in ths:
        tpr = float(np.mean([x >= t for x in pos_sims])) if pos_sims else 0.0
        fpr = float(np.mean([x >= t for x in neg_sims])) if neg_sims else 0.0
        roc.append((t, tpr, fpr))

    TARGET_FPR_STRICT = 0.001  # 0.1%
    TARGET_FPR_LOOSE = 0.01   # 1%
    candidates_strict = [t for (t, tpr, fpr) in roc if fpr <= TARGET_FPR_STRICT]
    candidates_loose = [t for (t, tpr, fpr) in roc if fpr <= TARGET_FPR_LOOSE]
    # For maximum detection power under the same FPR constraint, use the MIN threshold satisfying FPR
    tau_strict_min = min(candidates_strict) if candidates_strict else None
    tau_loose_min = min(candidates_loose) if candidates_loose else None
    # For ultra-conservative setting (lowest FPR safety margin), use the MAX threshold
    tau_strict_max = max(candidates_strict) if candidates_strict else None
    tau_loose_max = max(candidates_loose) if candidates_loose else None

    print('Samples:', len(scores))
    print('ROC (first 5):', roc[:5])
    print('tau FPR<=0.1%  (min -> more TPR):', tau_strict_min, '| (max -> more margin):', tau_strict_max)
    print('tau FPR<=1%    (min -> more TPR):', tau_loose_min,  '| (max -> more margin):', tau_loose_max)

    # Small summary per-attack
    per_attack = {}
    for atk in attacks.keys():
        sims = [s for (_, a, s, _) in scores if a == atk]
        if sims:
            per_attack[atk] = (float(np.mean(sims)), float(np.std(sims)))
    print('Per-attack mean±std sim:', per_attack)


if __name__ == '__main__':
    main()
