import cv2
import numpy as np
import pywt

def embedding(input1, input2="ecorp.npy"):
    """
    High-WPSNR and robust watermark embedding (v3).
    Target: 60–61 dB average WPSNR with solid robustness.
    """

    # ===== PARAMETERS =====
    FIXED_SEED = 42
    WATERMARK_SIZE = 1024

    # tuned embedding strengths
    alpha_L3 = 1.4   # high-freq bands (HL3, LH3)
    alpha_L2 = 2.2   # mid-freq bands (HL2, LH2)
    TOP_BLOCKS = 150 # fewer modified blocks per band

    # ===== I/O =====
    I = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if I is None:
        raise ValueError(f"Cannot read input image: {input1}")
    I = I.astype(np.float32)

    # Load or generate watermark bits
    if input2 is None:
        rng = np.random.default_rng(FIXED_SEED)
        Wbits = rng.integers(0, 2, size=WATERMARK_SIZE, dtype=np.uint8)
        np.save("generated_watermark.npy", Wbits)
    else:
        Wbits = np.load(input2).astype(np.uint8)
        if Wbits.size != WATERMARK_SIZE:
            raise ValueError(f"Watermark must be {WATERMARK_SIZE} bits.")

    # ===== DWT 3 levels =====
    coeffs = pywt.wavedec2(I, wavelet='db2', level=3)
    (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs[1], coeffs[2], coeffs[3]
    bands = [HL3, LH3, HL2, LH2]

    # ===== Define masks =====
    mask_L3 = [(1,2), (2,1), (2,2), (1,3), (3,1)]             # 5 coeffs on L3
    mask_L2 = [(1,0), (0,1), (1,1), (2,1), (1,2)]             # 5 coeffs on L2

    # ===== PN sequences =====
    rng = np.random.default_rng(FIXED_SEED)
    PN0_L3 = rng.standard_normal(len(mask_L3))
    PN1_L3 = rng.standard_normal(len(mask_L3))
    PN0_L2 = rng.standard_normal(len(mask_L2))
    PN1_L2 = rng.standard_normal(len(mask_L2))

    # ===== Helper: top-variance block selection =====
    def top_variance_positions(B, block_size=4, nblocks=TOP_BLOCKS):
        h, w = B.shape
        h_eff, w_eff = min(64, h), min(64, w)
        positions, scores = [], []
        for by in range(0, h_eff, block_size):
            for bx in range(0, w_eff, block_size):
                blk = B[by:by+block_size, bx:bx+block_size]
                positions.append((by, bx))
                scores.append(np.var(blk))
        idx = np.argsort(scores)[::-1][:nblocks]
        return [positions[i] for i in idx]

    pos_bands = [top_variance_positions(B, 4, TOP_BLOCKS) for B in bands]

    # ===== Embedding =====
    idx = 0
    for b in range(4):
        B = bands[b]
        if b <= 1:  # HL3, LH3
            mask, alpha, PN0, PN1 = mask_L3, alpha_L3, PN0_L3, PN1_L3
        else:       # HL2, LH2
            mask, alpha, PN0, PN1 = mask_L2, alpha_L2, PN0_L2, PN1_L2

        for (by, bx) in pos_bands[b]:
            if idx >= WATERMARK_SIZE:
                break
            block = B[by:by+4, bx:bx+4].copy()
            C = cv2.dct(block)

            # stronger attenuation on very noisy blocks
            v = np.var(block)
            alpha_local = alpha * (1 - 0.4 * min(v / 400.0, 1.0))

            bit = int(Wbits[idx])
            PN = PN1 if bit == 1 else PN0
            for k, (u, v) in enumerate(mask):
                C[u, v] += alpha_local * PN[k]
            B[by:by+4, bx:bx+4] = cv2.idct(C)
            idx += 1

        bands[b] = B

    # ===== Reconstruct =====
    new_coeffs = list(coeffs)
    LH3n, HL3n = bands[1], bands[0]
    LH2n, HL2n = bands[3], bands[2]
    new_coeffs[1] = (LH3n, HL3n, HH3)
    new_coeffs[2] = (LH2n, HL2n, HH2)
    new_coeffs[3] = (LH1, HL1, HH1)

    Iw = pywt.waverec2(new_coeffs, wavelet='db2')
    Iw = np.clip(Iw, 0, 255).astype(np.uint8)

    # ===== Return result =====
    return Iw
