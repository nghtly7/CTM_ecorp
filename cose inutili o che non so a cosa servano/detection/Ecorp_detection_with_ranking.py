import os
import pickle
import numpy as np
import cv2
from hashlib import sha256

# Se usi pywt nell'embedding, la detection può usarlo per DWT; altrimenti fallback
try:
    import pywt
except Exception:
    pywt = None

# ---------- UTILITIES (ricopiati / compatibili con embedding) ----------

def _hanning2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx).astype(np.float32)

def _compute_logpolar_fft_template(img_u8: np.ndarray, out_size=(360, 200)) -> np.ndarray:
    """Stessa funzione usata in embedding: log-polar del log-magnitude dello spettro."""
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
    return lp.astype(np.float32)

def _dwt2_level2(img_f32: np.ndarray):
    if pywt is None:
        return {"LL2": img_f32, "LH2": None, "HL2": None, "HH2": None, "_fallback": True}
    wavelet = "bior4.4"
    coeffs1 = pywt.dwt2(img_f32, wavelet)
    (LL1, (LH1, HL1, HH1)) = coeffs1
    coeffs2 = pywt.dwt2(LL1, wavelet)
    (LL2, (LH2, HL2, HH2)) = coeffs2
    return {"LL2": LL2, "LH2": LH2, "HL2": HL2, "HH2": HH2, "L1": (LH1, HL1, HH1), "wavelet": wavelet, "_fallback": False}

def _idwt2_level2(parts) -> np.ndarray:
    if pywt is None or parts.get("_fallback", False):
        return parts["LL2"].astype(np.float32)
    wavelet = parts["wavelet"]
    LL2 = parts["LL2"]
    LH2 = parts["LH2"]
    HL2 = parts["HL2"]
    HH2 = parts["HH2"]
    LH1, HL1, HH1 = parts["L1"]
    LL1 = pywt.idwt2((LL2, (LH2, HL2, HH2)), wavelet)
    img = pywt.idwt2((LL1, (LH1, HL1, HH1)), wavelet)
    return img.astype(np.float32)

def _block_dct(img: np.ndarray) -> np.ndarray:
    H, W = img.shape
    H8, W8 = H // 8 * 8, W // 8 * 8
    imgc = img[:H8, :W8].astype(np.float32)
    dct = np.zeros_like(imgc, dtype=np.float32)
    for y in range(0, H8, 8):
        for x in range(0, W8, 8):
            patch = imgc[y:y + 8, x:x + 8]
            dct[y:y + 8, x:x + 8] = cv2.dct(patch)
    # se serve pad later, manteniamo shape H8xW8
    return dct

def _block_idct(dct: np.ndarray, H: int, W: int) -> np.ndarray:
    H8, W8 = dct.shape
    img = np.zeros((H8, W8), dtype=np.float32)
    for y in range(0, H8, 8):
        for x in range(0, W8, 8):
            patch = dct[y:y + 8, x:x + 8]
            img[y:y + 8, x:x + 8] = cv2.idct(patch)
    if H8 != H or W8 != W:
        out = np.zeros((H, W), dtype=np.float32)
        out[:H8, :W8] = img
        return out
    return img

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

def _alpha_matrix(alpha0: float = 0.008, beta: float = 0.6) -> np.ndarray:
    a = np.zeros((8, 8), dtype=np.float32)
    for u in range(8):
        for v in range(8):
            freq = np.sqrt(float(u * u + v * v))
            csf = np.exp(-0.25 * (freq / 4.5) ** 2)
            a[u, v] = alpha0 * (1.0 - csf) + 0.001
    return a

def _block_variance_map(img: np.ndarray, block: int = 8) -> np.ndarray:
    H, W = img.shape
    h = (H // block) * block
    w = (W // block) * block
    imgc = img[:h, :w]
    by = h // block
    bx = w // block
    var_map = np.zeros((by, bx), dtype=np.float32)
    for iy in range(by):
        for ix in range(bx):
            patch = imgc[iy * block:(iy + 1) * block, ix * block:(ix + 1) * block]
            var_map[iy, ix] = np.var(patch.astype(np.float32))
    vmin, vmax = np.percentile(var_map, [5, 95])
    if vmax > vmin:
        var_norm = np.clip((var_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    else:
        var_norm = np.zeros_like(var_map)
    return 0.8 + 0.4 * var_norm

# ---------- Hamming(15,11) decoder (inverso dell'encoder che hai mostrato) ----------
def _ecc_decode_hamming1511(encoded_bits: np.ndarray, k=11, n=15):
    """Decode Hamming(15,11) with single-bit correction.
       encoded_bits: 1D array length multiple of n (0/1)
       Returns: decoded_bits (before removing pad) and number of corrected errors and uncorrectable (should be 0).
    """
    encoded_bits = np.asarray(encoded_bits, dtype=np.uint8).reshape(-1)
    if encoded_bits.size % n != 0:
        raise ValueError("encoded_bits length must be multiple of n")
    blocks = encoded_bits.size // n
    data_positions = np.array([3,5,6,7,9,10,11,12,13,14,15]) - 1  # 0-indexed
    parity_positions = np.array([1,2,4,8]) - 1
    decoded = []
    corrected = 0
    uncorrectable = 0
    for b in range(blocks):
        cw = encoded_bits[b*n:(b+1)*n].copy()
        # compute syndrome (parity bits over positions)
        # parity check for positions p in parity_positions: parity covers all positions where index & (p+1) != 0
        syndrome = 0
        for i, p in enumerate(parity_positions):
            mask = (((np.arange(1, n+1) & (p+1)) != 0).astype(np.uint8))
            # include parity bit in mask when computing syndrome parity (we want sum %2)
            parity = (np.sum(cw[mask.astype(bool)]) % 2)
            if parity != 0:
                syndrome |= (1 << i)
        if syndrome != 0:
            # syndrome gives the 1-indexed position of error if <= n
            errpos = syndrome - 1
            if 0 <= errpos < n:
                cw[errpos] ^= 1
                corrected += 1
            else:
                uncorrectable += 1
        # extract data bits
        data = cw[data_positions]
        decoded.append(data)
    decoded_bits = np.concatenate(decoded).astype(np.uint8)
    return decoded_bits, int(corrected), int(uncorrectable)

# ---------- DETECTION PRINCIPALE ----------

def detection(state_path: str, attacked_image_path: str, verbose: bool = True):
    """
    Effettua la detection dell'acqua digitale.
    - state_path: percorso a state.pkl salvato dall'embedding
    - attacked_image_path: immagine (grayscale o RGB) su cui effettuare detection
    Ritorna: dict con chiavi 'recovered_bits', 'orig_bits', 'ecc_report', 'BER' (se orig_bits disponibile)
    """
    # 1) carica stato
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    LP_ref = state["LP_ref"].astype(np.float32)
    M_embed = np.asarray(state["M_embed"], dtype=np.int32)  # shape (usable,5)
    seed = int(state["seed"])
    ecc_params = state["ecc_params"]
    alpha_matrix = state.get("alpha_matrix", _alpha_matrix())
    R = int(state["R"])
    zigzag_range = tuple(state.get("zigzag_range", (10, 30)))
    subband_names = state["subbands"]
    dims = state["dims"]
    orig_bits = int(ecc_params.get("orig_bits", 0))
    n = int(ecc_params["n"]); k = int(ecc_params["k"]); pad = int(ecc_params.get("pad", 0))

    # 2) leggi immagine attaccata e converti grayscale 512x512
    I_att = cv2.imread(attacked_image_path, cv2.IMREAD_GRAYSCALE)
    if I_att is None:
        raise FileNotFoundError("Immagine attaccata non trovata.")
    I_att = cv2.resize(I_att, (512, 512), interpolation=cv2.INTER_AREA)
    I_att_f32 = I_att.astype(np.float32) / 255.0

    # 3) calcola LP template dell'immagine attaccata e trova shift tramite phase correlation
    LP_att = _compute_logpolar_fft_template(I_att, out_size=LP_ref.shape)
    # cv2.phaseCorrelate richiede float32
    shift = cv2.phaseCorrelate(np.float32(LP_ref), np.float32(LP_att))[0]  # returns (dx,dy)
    dx, dy = float(shift[0]), float(shift[1])
    # Converti shift in rotazione e scala (usando la stessa M usata per creare LP)
    h, w = I_att.shape
    center = (w/2.0, h/2.0)
    r_max = np.hypot(center[0], center[1])
    M_lp = LP_ref.shape[0] / np.log(r_max + 1e-6)
    # shift in x corrisponde ad angolo (360 deg across width)
    angle_deg = - (dx / LP_ref.shape[1]) * 360.0
    # shift in y corrisponde a log scale
    scale = np.exp(- (dy / M_lp))

    if verbose:
        print(f"[detect] phaseCorrelate shift dx={dx:.3f}, dy={dy:.3f} -> angle={angle_deg:.3f} deg, scale={scale:.5f}")

    # 4) applica trasformazione inversa per riallineare (ruota di angle_deg e scala)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    I_corr_u8 = cv2.warpAffine((I_att_f32 * 255.0).astype(np.uint8), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    I_corr = I_corr_u8.astype(np.float32) / 255.0

    # 5) calcola DWT level2 sui dati riallineati (come in embedding) e prendi le stesse sottobande
    parts = _dwt2_level2(I_corr)
    fallback = parts.get("_fallback", False)
    subbands = []
    if not fallback:
        for name in ("LH2", "HL2"):
            sb = parts.get(name, None)
            if sb is not None:
                subbands.append((name, sb))
    else:
        subbands.append(("LL2", parts["LL2"]))

    # 6) ricostruisci la stessa logica di selezione delle posizioni (all_positions)
    # (usiamo zigzag_range e sel_positions)
    zigzag = _zigzag_indices_8x8()
    sel_mask = np.zeros((8, 8), dtype=bool)
    for kidx, (u, v) in enumerate(zigzag):
        if zigzag_range[0] <= kidx <= zigzag_range[1]:
            sel_mask[u, v] = True
    sel_positions = np.argwhere(sel_mask)

    # ricrea mappe varianza per subband
    var_maps = []
    for (_, sb_arr) in subbands:
        var_maps.append(_block_variance_map(sb_arr, block=8))

    # RICREARE robustness_map_64: per farlo dobbiamo simulare gli stessi attacchi come in embedding.
    # Per semplicità ricostruiamo robustness_map con lo stesso metodo _compute_robustness_mask usato in embedding.
    # Implementazione rapida: simulate gli attacchi con le stesse funzioni (se presenti).
    def _simulate_attack_diff_simple(I_f32):
        # usa una piccola collezione di attacchi equivalenti all'embedding
        from scipy.ndimage import gaussian_filter
        from scipy.signal import medfilt
        diffs = np.zeros_like(I_f32, dtype=np.float32)
        # blur
        diffs += np.abs((gaussian_filter(I_f32, 0.5) - I_f32))
        # median
        try:
            diffs += np.abs((medfilt((I_f32*255.0).astype(np.uint8), 5).astype(np.float32)/255.0 - I_f32))
        except Exception:
            pass
        # AWGN std ~5/255
        diffs += np.abs(((np.clip(I_f32 + np.random.normal(0, 5.0/255.0, I_f32.shape), 0, 1)) - I_f32))
        # sharpening
        filter_blurred_f = gaussian_filter(I_f32, 2)
        diffs += np.abs(I_f32 + 1 * (I_f32 - filter_blurred_f) - I_f32)
        # resizing (down-up)
        try:
            from skimage.transform import rescale
            r = rescale(I_f32, 0.75, anti_aliasing=True)
            r2 = rescale(r, 1/0.75, anti_aliasing=True)
            r2 = r2[:I_f32.shape[0], :I_f32.shape[1]]
            diffs += np.abs(r2 - I_f32)
        except Exception:
            pass
        return diffs

    total_diff = _simulate_attack_diff_simple(I_corr)
    # compute attack_map 64x64 like embedding does -> block size 8 on 512 image
    H, W = I_corr.shape
    H8, W8 = H // 8 * 8, W // 8 * 8
    by, bx = H8 // 8, W8 // 8
    attack_map = np.zeros((by, bx), dtype=np.float32)
    for iy in range(by):
        for ix in range(bx):
            patch = total_diff[iy*8:(iy+1)*8, ix*8:(ix+1)*8]
            attack_map[iy, ix] = np.mean(patch)
    vmin, vmax = np.percentile(attack_map, [5, 95])
    attack_norm = np.clip((attack_map - vmin) / (vmax - vmin + 1e-8), 0, 1) if vmax>vmin else np.zeros_like(attack_map)
    robustness_map_64 = attack_norm  # 64x64 as in embedding

    # ora downscale robustness_map_64 in maniera identica a embedding (r_map_sb = robustness_map_64[::2, ::2])
    # e costruisci all_positions (nell'ordine definito nell'embedding)
    all_positions = []
    sb_arrays = []
    sb_shapes = []
    for si, (sb_name, sb_arr) in enumerate(subbands):
        Hs, Ws = sb_arr.shape
        H8s, W8s = Hs // 8 * 8, Ws // 8 * 8
        by_s, bx_s = H8s // 8, W8s // 8
        sb_arrays.append(sb_arr)
        sb_shapes.append((Hs, Ws, by_s, bx_s))
        # downscale mapping come nell'embedding
        r_map_sb = robustness_map_64[::2, ::2]
        for iy in range(by_s):
            for ix in range(bx_s):
                # aggiungi posizioni mid-frequency nel blocco
                for (u, v) in sel_positions:
                    all_positions.append((si, iy, ix, int(u), int(v)))
    total_positions = len(all_positions)
    if verbose:
        print(f"[detect] total_positions ricostruite = {total_positions}, M_embed length = {len(M_embed)}")

    # 7) ricostruisci DCT subbands come in embedding
    dct_subbands = []
    for (_, sb_arr) in subbands:
        dct_subbands.append(_block_dct(sb_arr))

    # 8) ricostruisci RNG e permutazioni nello stesso ordine per ottenere la sequenza di segni 's'
    # compute encoded payload length (len(payload) in embedding)
    # blocks = (orig_bits + pad)/k
    blocks_count = (orig_bits + pad) // k
    encoded_len = blocks_count * n  # len(payload)
    if encoded_len <= 0:
        raise RuntimeError("Encoded payload length ricostruita = 0; controllo ecc_params.")

    rng = np.random.RandomState(seed)
    # same calls order as embedding:
    _ = rng.permutation(encoded_len)            # perm_idx = rng.permutation(len(payload))
    _ = rng.permutation(total_positions)        # pos permutation in embedding
    # now RNG is in the same state as prima di generare s values
    # generate R*encoded_len s values (embedding generated rng.rand() per used coefficient in that order)
    usable = M_embed.shape[0]
    s_seq = (rng.rand(usable) > 0.5).astype(np.int8)  # True->1 else 0
    s_seq = np.where(s_seq > 0, 1.0, -1.0).astype(np.float32)

    # 9) leggere i coefficienti secondo l'ordine M_embed e produrre un segno per ogni coeff
    coeff_signs = np.zeros(usable, dtype=np.float32)
    for i_pos, (si, byb, bxb, u, v) in enumerate(M_embed):
        si = int(si); byb=int(byb); bxb=int(bxb); u=int(u); v=int(v)
        dct_arr = dct_subbands[si]
        y0 = byb * 8
        x0 = bxb * 8
        # prendi coefficiente (assumiamo che dct_subbands[si] abbia formato H8xW8)
        c = dct_arr[y0 + u, x0 + v]
        coeff_signs[i_pos] = 1.0 if c >= 0 else -1.0  # segno del coeff
    # Correggi il segno con s_seq: in embedding si aggiungeva delta = alpha*bit_sgn*s*mask_scale
    evidence = coeff_signs * s_seq  # +1 -> prob bit +1, -1 -> prob bit -1

    # 10) aggrega repliche R per ogni payload bit (payload_perm order)
    if usable % R != 0:
        # attenzione: qualcosa non quadra; prosegui comunque con floor
        n_payload_perm = usable // R
    else:
        n_payload_perm = usable // R

    bit_vals_perm = np.zeros(n_payload_perm, dtype=np.int8)
    for i in range(n_payload_perm):
        seg = evidence[i*R:(i+1)*R]
        ssum = np.sum(seg)
        bit_vals_perm[i] = 1 if ssum >= 0 else 0

    # 11) inverti permutazione per ottenere payload originale (prima decode ECC)
    # ricostruisci la perm_idx con rng reset (deterministico)
    rng2 = np.random.RandomState(seed)
    perm_idx = rng2.permutation(encoded_len)
    # perm_idx permuta gli indici 0..encoded_len-1; embedding fece payload_perm = payload[perm_idx]
    # quindi payload_perm length == encoded_len e noi abbiamo bit_vals_perm che è la versione permutata
    # per invertire: payload = inverse_permutation(payload_perm)
    if len(perm_idx) != n_payload_perm:
        # in teoria perm_idx len = encoded_len, n_payload_perm = encoded_len -> devono combaciare
        # se non combaciano, qualcosa non quadra: tronchiamo o estendiamo
        minlen = min(len(perm_idx), len(bit_vals_perm))
        inv = np.zeros(minlen, dtype=np.uint8)
        inv_perm = np.argsort(perm_idx[:minlen])
        payload_recovered = bit_vals_perm[:minlen][inv_perm]
    else:
        inv_perm = np.argsort(perm_idx)
        payload_recovered = bit_vals_perm[inv_perm]

    # 12) ECC decode Hamming(15,11)
    decoded_bits, corrected, uncorrectable = _ecc_decode_hamming1511(payload_recovered.astype(np.uint8), k=k, n=n)
    # remove pad to get original length
    total_bits = decoded_bits.size
    if pad > 0:
        recovered_bits = decoded_bits[: (total_bits - pad)]
    else:
        recovered_bits = decoded_bits[:orig_bits] if orig_bits>0 else decoded_bits

    # 13) report
    report = {
        "recovered_bits": recovered_bits.astype(np.uint8),
        "orig_bits": orig_bits,
        "ecc": {"corrected": corrected, "uncorrectable": uncorrectable},
        "usable_positions": usable,
        "R": R,
    }
    if orig_bits > 0:
        # se l'utente ha salvato l'originale dei bit (non sempre disponibile), possiamo calcolare BER
        # qui non abbiamo i bit originali veri a meno che li fornisci; settiamo BER a None
        report["BER"] = None

    if verbose:
        print(f"[detect] ECC: corrected={corrected}, uncorrectable={uncorrectable}, usable={usable}, R={R}")
        print(f"[detect] recovered bits length = {recovered_bits.size}, orig_bits (state) = {orig_bits}")

    return report

# ---------------- USO ----------------
# Esempio:
# report = detect("/path/to/state.pkl", "/path/to/attacked_image.png", verbose=True)
# bits = report["recovered_bits"]
