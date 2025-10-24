import cv2
import numpy as np
import pywt
import os
from wpsnr import wpsnr as WPSNR


# ---- Attack-map helpers ----
def _awgn(img_f32, sigma, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img_f32.shape).astype(np.float32)
    out = img_f32 + noise
    return np.clip(out, 0, 255)

def _gblur(img_u8, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))  # kernel dispari ~ 6*sigma
    return cv2.GaussianBlur(img_u8, (k, k), sigmaX=sigma)

def _median(img_u8, k):
    return cv2.medianBlur(img_u8, k)

def _resize_roundtrip(img_u8, scale):
    h, w = img_u8.shape[:2]
    small = cv2.resize(img_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    back  = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back

def compute_attack_map(I_u8):

    # Restituisce una mappa 0..1 (stessa size di I_u8) che stima la vulnerabilità locale:
    # somma |attacked - original| per attacchi leggeri, poi normalizza e fa smoothing.
    
    orig_u8 = I_u8
    orig_f  = orig_u8.astype(np.float32)
    A = np.zeros_like(orig_f, dtype=np.float32)

    # AWGN leggero
    for std in (0.5, 2.0, 5.0):
        att = _awgn(orig_f, std)
        A += np.abs(att - orig_f)

    # Blur gaussiano
    for s in (0.8, 1.5):
        att = _gblur(orig_u8, s).astype(np.float32)
        A += np.abs(att - orig_f)

    # Median filter
    for k in (3, 5):
        att = _median(orig_u8, k).astype(np.float32)
        A += np.abs(att - orig_f)

    # Resize round-trip
    for sc in (0.9, 0.75):
        att = _resize_roundtrip(orig_u8, sc).astype(np.float32)
        A += np.abs(att - orig_f)

    # Normalizzazione 0..1 + smoothing (NB: usare np.ptp con NumPy>=2.0)
    A = (A - A.min()) / (np.ptp(A) + 1e-12)
    A = cv2.GaussianBlur(A, (5, 5), 0.8)
    return A

# === GLOBAL PARAMS (used by tuning and normal embedding) ===
global beta, gamma, Q, soft_t, soft_k
beta   = 0.6
gamma  = 0.4
Q      = 0.6
soft_t = 0.15
soft_k = 0.6

# weaker default parameters
#Q      = 0.3
beta   = 0.3
soft_t = 0.30
soft_k = 1.0 # al momento nemmeno usato
band_scale = [1.0, 1.0, 0.7, 0.7]   # bande fini più forti, bande grossolane più deboli


def embedding(input1, input2='ecorp.npy'):
    # Parametri embedding
    alpha = 3.0     # seed iniziale (verrà riscalato)
    FIXED_SEED = 42

    # Target WPSNR preciso
    TARGET_WPSNR = 70.00
    TOL = 0.05
    MAX_ITERS = 4

    # 1) I/O
    I = cv2.imread(input1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if input2 is None:
        rng = np.random.default_rng(FIXED_SEED)
        Wbits = rng.integers(0, 2, size=1024, dtype=np.uint8)
    else:
        if not os.path.isabs(input2):
            input2 = os.path.join(os.path.dirname(__file__), "..", input2)
        Wbits = np.load(input2).astype(np.uint8)

    H, W = I.shape
    I_u8 = I.astype(np.uint8)

    # 2) DWT 3 livelli (coeffs di PARTENZA, usati ad ogni re-embed)
    coeffs0 = pywt.wavedec2(I, wavelet='db2', level=3)
    (LH3_0, HL3_0, HH3_0), (LH2_0, HL2_0, HH2_0), (LH1_0, HL1_0, HH1_0) = coeffs0[1], coeffs0[2], coeffs0[3]

    # 3) set mid-band e PN
    #    (5-coeff mask "weaker" come da tuo file)
    mask = [(0,1),(1,0),(1,1),(2,0),(0,2)]
    rng = np.random.default_rng(FIXED_SEED)
    PN0 = rng.standard_normal(len(mask)).astype(np.float32)
    PN1 = rng.standard_normal(len(mask)).astype(np.float32)
    PN0 /= (np.linalg.norm(PN0) + 1e-12)
    PN1 /= (np.linalg.norm(PN1) + 1e-12)

    # 4) Attack-map (usata nel calcolo alpha_block)
    attack_map = compute_attack_map(I_u8)

    # 5) PRE-CALCOLO ENERGIE GLOBALI su bande di partenza (per E_mean, E_ptp)
    #    (HL3, LH3, HL2, LH2) come da tuo schema
    def make_bands_from_coeffs(c):
        (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = c[1], c[2], c[3]
        return [HL3.copy(), LH3.copy(), HL2.copy(), LH2.copy()], (LH3, HL3, HH3, LH2, HL2, HH2, LH1, HL1, HH1)

    bands_init, _ = make_bands_from_coeffs(coeffs0)

    energies = []
    for b in range(4):
        B = bands_init[b]
        for by in range(0,64,4):
            for bx in range(0,64,4):
                C = cv2.dct(B[by:by+4, bx:bx+4].astype(np.float32))
                vals = np.array([C[u,v] for (u,v) in mask], dtype=np.float32)
                energies.append(np.linalg.norm(vals))
    energies = np.array(energies, dtype=np.float32)
    E_mean = float(np.mean(energies))
    E_ptp  = float(np.ptp(energies) + 1e-12)

    # 6) DRY-RUN: stima MSE per scalare alpha su target (prima stima)
    alpha_blocks = []
    idx = 0
    # NB: uso i tuoi global già presenti nel file
    global beta, gamma, soft_t, soft_k, Q, band_scale
    # nel dry-run non usiamo gamma/attack_map (serve solo la stima energetica)
    for b in range(4):
        B = bands_init[b]
        for by in range(0, 64, 4):
            for bx in range(0, 64, 4):
                block = B[by:by+4, bx:bx+4].astype(np.float32)
                C = cv2.dct(block)
                vals = np.array([C[u,v] for (u,v) in mask], dtype=np.float32)
                E = np.linalg.norm(vals)
                E_norm = (E - E_mean) / (E_ptp + 1e-12)
                local_mean = float(block.mean())
                lum_mask = 1.0 + 0.15 * (local_mean/128.0 - 1.0)

                # uso beta "weaker" come nel tuo file corrente
                beta_local = 0.3 if beta is None else beta
                alpha_block = alpha * (1.0 + beta_local * E_norm) * lum_mask
                alpha_block *= band_scale[b] if 'band_scale' in globals() else 1.0
                alpha_block = float(np.clip(alpha_block, 0.1*alpha, 2.0*alpha))
                alpha_blocks.append(alpha_block)
                idx += 1

    pred_MSE = float(np.sum(np.array(alpha_blocks, dtype=np.float32)**2) / (H * W))
    target_mse = (255.0**2) / (10**(TARGET_WPSNR/10))
    alpha *= np.sqrt(target_mse / (pred_MSE + 1e-12))  # prima stima

    # ---- funzione di EMBEDDING reale (reset & re-embed) ----
    def do_embed_with_alpha(alpha_curr: float) -> np.ndarray:
        # ricrea bands da coeffs0 (reset)
        bands, (LH3, HL3, HH3, LH2, HL2, HH2, LH1, HL1, HH1) = make_bands_from_coeffs(coeffs0)

        idx = 0
        for b in range(4):
            B = bands[b]
            for by in range(0, 64, 4):
                for bx in range(0, 64, 4):
                    block = B[by:by+4, bx:bx+4].astype(np.float32)
                    C = cv2.dct(block)

                    bit = int(Wbits[idx])
                    x = np.array([C[u,v] for (u,v) in mask], dtype=np.float32)
                    p = PN1 if bit==1 else PN0

                    # HIR: proietta PN ortogonale a x
                    den = (x @ x) + 1e-12
                    p_ortho = p - ((p @ x) / den) * x
                    p_use = p if np.linalg.norm(p_ortho) < 1e-6 else p_ortho / (np.linalg.norm(p_ortho) + 1e-12)

                    # feature locali
                    E = np.linalg.norm(x)
                    E_norm = (E - E_mean) / E_ptp
                    local_mean = float(block.mean())
                    lum_mask = 1.0 + 0.15 * (local_mean/128.0 - 1.0)

                    img_y = int((by/64.0) * H)
                    img_x = int((bx/64.0) * W)
                    h = max(1, H//64); w = max(1, W//64)
                    y0 = max(0, img_y - h//2); y1 = min(H, y0 + h)
                    x0 = max(0, img_x - w//2); x1 = min(W, x0 + w)
                    attack_score = float(attack_map[y0:y1, x0:x1].mean()) if (y1>y0 and x1>x0) else 0.0

                    # alpha per-blocco (usa i global correnti)
                    beta_local = beta if beta is not None else 0.3
                    gamma_local = gamma if gamma is not None else 0.4
                    alpha_block = alpha_curr * (1.0 + beta_local * E_norm) * (1.0 - gamma_local * attack_score) * lum_mask

                    # soft handling
                    if E < soft_t * E_mean:
                        idx += 1
                        continue  # skip totale dei blocchi piatti (come nel tuo file attuale)

                    # band scaling se presente
                    if 'band_scale' in globals():
                        alpha_block *= band_scale[b]

                    alpha_block = float(np.clip(alpha_block, 0.1*alpha_curr, 2.0*alpha_curr))

                    # iniezione watermark
                    for k,(u,v) in enumerate(mask):
                        C[u,v] += alpha_block * p_use[k]

                    # quantizzazione correttiva
                    for (u,v) in mask:
                        C[u,v] = Q * np.round(C[u,v] / Q)

                    B[by:by+4, bx:bx+4] = cv2.idct(C)
                    idx += 1

            bands[b] = B

        # ricostruzione coeffs e IDWT
        new_coeffs = list(coeffs0)
        LH3n, HL3n = bands[1], bands[0]
        LH2n, HL2n = bands[3], bands[2]
        new_coeffs[1] = (LH3n, HL3n, HH3)
        new_coeffs[2] = (LH2n, HL2n, HH2)
        new_coeffs[3] = (LH1,   HL1,   HH1)
        Iw = pywt.waverec2(new_coeffs, wavelet='db2')
        Iw = np.clip(Iw, 0, 255).astype(np.uint8)
        return Iw

    # 7) REFINEMENT loop su WPSNR reale (senza stampe)
    Iw_best = None
    for _ in range(MAX_ITERS):
        Iw = do_embed_with_alpha(alpha)
        w_real = float(WPSNR(I_u8, Iw))

        if abs(w_real - TARGET_WPSNR) <= TOL:
            Iw_best = Iw
            break

        curr_mse = (255.0**2) / (10**(w_real/10))
        alpha *= np.sqrt(target_mse / (curr_mse + 1e-12))
        Iw_best = Iw  # in caso non converga, restituisco l'ultimo

    if Iw_best is None:
        Iw_best = do_embed_with_alpha(alpha)

    return Iw_best
