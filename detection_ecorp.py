import os
import sys
import cv2
import numpy as np


# Setup paths to import wpsnr and locate state.pkl
_THIS_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.dirname(_THIS_DIR)  # CTM_ecorp
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from wpsnr import wpsnr as compute_wpsnr
except Exception:
    compute_wpsnr = None

try:
    import pywt  # type: ignore
except Exception:
    pywt = None


# ------------------ Helpers ------------------

def _read_gray_512(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return img


def _hanning2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx).astype(np.float32)


def _logpolar_fft(img_u8: np.ndarray, out_size=(360, 200)) -> tuple[np.ndarray, float]:
    h, w = img_u8.shape
    win = _hanning2d(h, w)
    img = img_u8.astype(np.float32) / 255.0
    imgw = img * win
    F = np.fft.fftshift(np.fft.fft2(imgw))
    mag = np.log1p(np.abs(F)).astype(np.float32)
    if mag.max() > 0:
        mag = mag / (mag.max() + 1e-8)
    center = (w / 2.0, h / 2.0)
    r_max = np.hypot(center[0], center[1])
    M = out_size[0] / np.log(r_max + 1e-6)
    lp = cv2.logPolar(mag, center, M, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    if lp.shape != out_size:
        lp = cv2.resize(lp, (out_size[1], out_size[0]), interpolation=cv2.INTER_AREA)
    return lp.astype(np.float32), float(M)


def _phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    # a, b are float32 same size
    win = _hanning2d(*a.shape)
    shift, response = cv2.phaseCorrelate(a, b, win)
    # cv2 returns (dx, dy)
    return float(shift[0]), float(shift[1]), float(response)


def _zigzag_indices_8x8() -> np.ndarray:
    idx = [
        (0, 0),
        (0, 1), (1, 0),
        (2, 0), (1, 1), (0, 2),
        (0, 3), (1, 2), (2, 1), (3, 0),
        (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
        (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0),
        (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6),
        (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0),
        (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7),
        (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2),
        (7, 3), (6, 4), (5, 5), (4, 6), (3, 7),
        (4, 7), (5, 6), (6, 5), (7, 4),
        (7, 5), (6, 6), (5, 7),
        (6, 7), (7, 6),
        (7, 7),
    ]
    return np.array(idx, dtype=np.int32)


def _alpha_matrix(alpha0: float = 0.008) -> np.ndarray:
    a = np.zeros((8, 8), dtype=np.float32)
    for u in range(8):
        for v in range(8):
            freq = np.sqrt(float(u * u + v * v))
            csf = np.exp(-0.25 * (freq / 4.5) ** 2)
            a[u, v] = alpha0 * (1.0 - csf) + 0.001
    return a


def _dwt2_level2(img_f32: np.ndarray):
    if pywt is None:
        return {"LL2": img_f32, "LH2": None, "HL2": None, "HH2": None, "_fallback": True}
    wavelet = "bior4.4"
    LL1, (LH1, HL1, HH1) = pywt.dwt2(img_f32, wavelet)
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, wavelet)
    return {"LL2": LL2, "LH2": LH2, "HL2": HL2, "HH2": HH2, "L1": (LH1, HL1, HH1), "wavelet": wavelet, "_fallback": False}


def _block_dct(img: np.ndarray) -> np.ndarray:
    H, W = img.shape
    H8, W8 = H // 8 * 8, W // 8 * 8
    img = img[:H8, :W8].astype(np.float32)
    dct = np.zeros_like(img, dtype=np.float32)
    for y in range(0, H8, 8):
        for x in range(0, W8, 8):
            patch = img[y:y + 8, x:x + 8]
            dct[y:y + 8, x:x + 8] = cv2.dct(patch)
    return dct


def _ecc_decode_hamming1511(bits: np.ndarray, pad: int, orig_len: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n = 15
    k = 11
    if bits.size % n != 0:
        # truncate if misaligned
        L = bits.size // n * n
        bits = bits[:L]
    cw = bits.reshape(-1, n)
    parity_positions = np.array([1, 2, 4, 8], dtype=np.int32)
    data_positions = np.array([3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32)

    # Compute syndrome and correct single-bit errors
    corrected = cw.copy()
    idxs = np.arange(1, n + 1)
    for i in range(corrected.shape[0]):
        s = 0
        for p in parity_positions:
            mask = ((idxs & p) != 0)
            parity = int(np.sum(corrected[i, mask]) % 2)
            if parity != 0:
                s += p
        if 1 <= s <= n:
            corrected[i, s - 1] ^= 1
    data = corrected[:, data_positions - 1].reshape(-1)
    # Remove padding at the tail of the original k*blocks bits
    total_data_bits = data.size
    # The original data length after padding was ceil(orig_len/k)*k
    out_len = int(np.ceil(orig_len / k) * k)
    if out_len > total_data_bits:
        out_len = total_data_bits
    data = data[:out_len]
    # Remove pad to return exactly orig_len
    if pad > 0:
        data = data[:-pad]
    if data.size > orig_len:
        data = data[:orig_len]
    return data.astype(np.uint8)


def _build_positions_and_dcts(img_u8: np.ndarray, subbands_names: list[str]):
    img_f32 = img_u8.astype(np.float32) / 255.0
    parts = _dwt2_level2(img_f32)
    subbands = []
    if parts.get("_fallback", False):
        subbands = [("LL2", parts["LL2"])]
    else:
        for name in subbands_names:
            subbands.append((name, parts[name]))

    zigzag = _zigzag_indices_8x8()
    sel_mask = np.zeros((8, 8), dtype=bool)
    for k, (u, v) in enumerate(zigzag):
        if 10 <= k <= 30:
            sel_mask[u, v] = True
    sel_positions = np.argwhere(sel_mask)

    all_positions = []  # (si, by, bx, u, v)
    dct_subbands = []
    for si, (_, sb) in enumerate(subbands):
        H, W = sb.shape
        H8, W8 = H // 8 * 8, W // 8 * 8
        dct_arr = _block_dct(sb)
        dct_subbands.append(dct_arr)
        by, bx = H8 // 8, W8 // 8
        for iy in range(by):
            for ix in range(bx):
                for (u, v) in sel_positions:
                    all_positions.append((si, iy, ix, int(u), int(v)))

    return all_positions, dct_subbands


def _extract_bits_from_image(img_u8: np.ndarray, state: dict) -> np.ndarray:
    # Prepare DCT coefficients and positions
    # This function now expects reconstruction params passed via closure; see detection() usage below
    raise RuntimeError("_extract_bits_from_image should not be called directly in this refactor.")


def _align_fourier_mellin(att_u8: np.ndarray, LP_ref: np.ndarray) -> tuple[np.ndarray, float]:
    # Compute log-polar FFT magnitude of the attacked image
    LP_att, M = _logpolar_fft(att_u8, out_size=LP_ref.shape)
    dx, dy, resp = _phase_correlation_shift(LP_ref.astype(np.float32), LP_att.astype(np.float32))
    # Derive rotation and scale
    cols = LP_ref.shape[1]
    rot_deg = -dx * 360.0 / float(cols)
    scale = float(np.exp(dy / M))
    # Warp attacked to align
    center = (att_u8.shape[1] / 2.0, att_u8.shape[0] / 2.0)
    Maff = cv2.getRotationMatrix2D(center, rot_deg, 1.0 / max(scale, 1e-6))
    aligned = cv2.warpAffine(att_u8, Maff, (att_u8.shape[1], att_u8.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned, resp


def detection(input1, input2, input3):
    # Inputs: original (for WPSNR computation only), watermarked reference, attacked image
    orig_path = str(input1)
    wm_ref_path = str(input2)
    att_path = str(input3)

    I_wm = _read_gray_512(wm_ref_path)
    I_att = _read_gray_512(att_path)

    # 1) Build LP_ref from watermarked image (non-blind detection)
    LP_ref, M_lp = _logpolar_fft(I_wm, out_size=(360, 200))
    # Align attacked via Fourier–Mellin, but skip if transform is near identity
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

    # 2) Reconstruct embedding mapping deterministically
    # Subbands used: DWT available -> ["LH2","HL2"], else ["LL2"]
    subbands_names = ["LL2"] if pywt is None else ["LH2", "HL2"]

    def build_positions(img_u8: np.ndarray):
        img_f32 = img_u8.astype(np.float32) / 255.0
        parts = _dwt2_level2(img_f32)
        subs = []
        if parts.get("_fallback", False):
            subs = [("LL2", parts["LL2"])]
        else:
            for nm in subbands_names:
                subs.append((nm, parts[nm]))
        zigzag = _zigzag_indices_8x8()
        sel_mask = np.zeros((8, 8), dtype=bool)
        for k, (u, v) in enumerate(zigzag):
            if 10 <= k <= 30:
                sel_mask[u, v] = True
        sel_positions = np.argwhere(sel_mask)
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

    all_positions_ref, dct_subs_ref = build_positions(I_wm)
    total_positions = len(all_positions_ref)

    # ECC parameters (fixed by design)
    orig_len = 1024
    k, n = 11, 15
    blocks = int(np.ceil(orig_len / k))
    pad = blocks * k - orig_len
    L_ecc = blocks * n

    # Replication factor and usable positions matching embedding policy
    R = max(1, min(32, total_positions // L_ecc))
    usable = R * L_ecc

    # Seed derived from original image basename (must match embedding)
    base_name = os.path.basename(orig_path).encode("utf-8")
    from hashlib import sha256
    seed = int.from_bytes(sha256(base_name).digest()[:8], "big", signed=False) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)

    # Permutation of payload bits and selection of positions
    perm_idx = rng.permutation(L_ecc)
    pos_idx = rng.permutation(total_positions)[:usable]
    # Freeze the exact used coordinate tuples from reference image ordering
    used_positions = [all_positions_ref[i] for i in pos_idx]

    alpha_mat = _alpha_matrix(0.008)
    # Precompute random sign sequence ONCE for all used positions to keep extraction consistent
    sign_seq = np.where(rng.rand(usable) > 0.5, 1.0, -1.0).astype(np.float32)

    def extract_with_positions(img_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Build only DCT arrays for current image; use reference 'used_positions' for indexing
        _, dct_subs = build_positions(img_u8)
        sums = np.zeros(L_ecc, dtype=np.float64)
        for i in range(usable):
            # Use precomputed random sign per coefficient (consistent across extractions)
            sgn = float(sign_seq[i])
            si, by, bx, u, v = used_positions[i]
            # Clamp subband index if mismatch
            si_use = int(si)
            if si_use < 0 or si_use >= len(dct_subs):
                si_use = 0
            dct_arr = dct_subs[si_use]
            y0 = by * 8 + u
            x0 = bx * 8 + v
            if y0 >= dct_arr.shape[0] or x0 >= dct_arr.shape[1]:
                coeff = 0.0
            else:
                coeff = float(dct_arr[y0, x0])
            a = float(alpha_mat[u, v]) or 1.0
            norm_coeff = (coeff / a) * sgn
            bi = i // R
            sums[bi] += norm_coeff
        bits_perm = (sums > 0).astype(np.uint8)
        bits_ecc = np.zeros_like(bits_perm)
        bits_ecc[perm_idx] = bits_perm
        bits_dec = _ecc_decode_hamming1511(bits_ecc, pad=pad, orig_len=orig_len)
        return bits_dec, bits_ecc

    # Extract from reference watermarked and aligned attacked
    bits_ref_dec, bits_ref_ecc = extract_with_positions(I_wm)
    bits_att_dec, bits_att_ecc = extract_with_positions(aligned)

    # 3) Similarity and decision
    # Map ECC-coded bitstreams to +/-1 and compute normalized correlation (more robust)
    a = (bits_att_ecc.astype(np.float32) * 2.0 - 1.0)
    b = (bits_ref_ecc.astype(np.float32) * 2.0 - 1.0)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0

    # Detection threshold (tunable). Default set from ROC to keep FPR ≤ 0.1%.
    # Can be overridden at runtime with environment variable DET_TAU.
    default_tau = 0.85
    tau = default_tau
    _tau_env = os.environ.get("DET_TAU")
    if _tau_env:
        try:
            tau = float(_tau_env)
        except Exception:
            tau = default_tau
    decision = 1 if sim >= tau else 0

    # 5) WPSNR(watermarked, attacked)
    if compute_wpsnr is not None:
        try:
            wpsnr_val = float(compute_wpsnr(I_wm, I_att))
        except Exception:
            wpsnr_val = 0.0
    else:
        wpsnr_val = 0.0

    return int(decision), float(wpsnr_val)
