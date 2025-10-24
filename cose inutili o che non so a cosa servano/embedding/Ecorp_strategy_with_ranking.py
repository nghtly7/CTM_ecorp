import os
import sys
import cv2
import pickle
import numpy as np
from hashlib import sha256
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt


# Attempt to import PyWavelets; fallback to None
try:
	import pywt  # type: ignore
except Exception:
	pywt = None  # type: ignore


# Add project root to path to import wpsnr
_THIS_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.dirname(_THIS_DIR)  # CTM_ecorp folder
if _ROOT_DIR not in sys.path:
	sys.path.insert(0, _ROOT_DIR)

try:
	from wpsnr import wpsnr as compute_wpsnr
except Exception:
	compute_wpsnr = None  # Optional: embedding does not require it
 
def jpeg_compression(img, QF):
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    attacked = cv2.imread('tmp.jpg', 0)
    os.remove('tmp.jpg')
    return attacked

def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked

def awgn(img, std, seed=None):
    mean = 0.0
    rng = np.random.RandomState(seed) if seed is not None else np.random
    attacked = img.astype(np.float32) + rng.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked.astype(img.dtype)


def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked

def median(img, kernel_size):
    attacked = medfilt(img, kernel_size)
    return attacked

def resizing(img, scale):
  from skimage.transform import rescale
  x, y = img.shape
  attacked = rescale(img, scale)
  attacked = rescale(attacked, 1/scale)
  attacked = attacked[:x, :y]
  return attacked
 
"""
def _simulate_attack_diff_dwt(I_f32: np.ndarray, attack_func, *args, **kwargs) -> np.ndarray:
    #Simula un attacco e calcola la differenza a livello di immagine spaziale.
    I_u8 = (I_f32 * 255.0).astype(np.uint8)
    I_attacked_u8 = attack_func(I_u8, *args, **kwargs)
    I_attacked_f32 = I_attacked_u8.astype(np.float32) / 255.0
    return np.abs(I_attacked_f32 - I_f32)
"""

def _simulate_attack_diff_dwt(I_f32: np.ndarray, attack_func, *args, **kwargs) -> np.ndarray:
    """Simula un attacco e calcola la differenza a livello di immagine spaziale.
    Normalizza correttamente l'output dell'attacco indipendentemente dal tipo restituito."""
    # I_f32 è in [0,1] float32
    I_u8 = (np.clip(I_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
    I_attacked = attack_func(I_u8, *args, **kwargs)

    # Normalizza attacked a uint8 coerente
    if I_attacked.dtype == np.uint8:
        I_attacked_u8 = I_attacked
    else:
        # Se float, cerchiamo di capire l'intervallo
        I_attacked = np.asarray(I_attacked)
        if I_attacked.max() <= 1.001:
            # probabile range [0,1]
            I_attacked_u8 = np.clip((I_attacked * 255.0 + 0.5), 0, 255).astype(np.uint8)
        else:
            # probabile range [0,255] float
            I_attacked_u8 = np.clip(I_attacked, 0, 255).astype(np.uint8)

    I_attacked_f32 = I_attacked_u8.astype(np.float32) / 255.0
    return np.abs(I_attacked_f32 - I_f32).astype(np.float32)



def _compute_robustness_mask(I_f32: np.ndarray, block_size: int = 8) -> dict:
    """
    Esegue attacchi sull'immagine e calcola la maschera di robustezza (blank_image)
    e una mappa di robustezza media per blocco (attack_map).
    """
    
    # Lista degli attacchi da simulare
    attack_params = [
    #(blur, 0.5),
    (median, 5),
    (awgn, 5, 42),            # aggiunto seed
    (sharpening, 2, 1),
    (resizing, 0.75),
]

    total_diff = np.zeros_like(I_f32, dtype=np.float32)

    for attack_func, *params in attack_params:
        total_diff += _simulate_attack_diff_dwt(I_f32, attack_func, *params)
    
    # 2. CALCOLO DELLA MAPPA DI ROBUSTEZZA A LIVELLO DI BLOCCO
    H, W = I_f32.shape
    H8, W8 = H // block_size * block_size, W // block_size * block_size
    by, bx = H8 // block_size, W8 // block_size

    attack_map = np.zeros((by, bx), dtype=np.float32)
    
    for iy in range(by):
        for ix in range(bx):
            patch_diff = total_diff[iy * block_size:(iy + 1) * block_size, ix * block_size:(ix + 1) * block_size]
            attack_map[iy, ix] = np.mean(patch_diff)

    # Normalizza l'attack_map per il ranking (0 = più robusto, 1 = meno robusto)
    vmin, vmax = np.percentile(attack_map, [5, 95])
    if vmax > vmin:
        attack_norm = np.clip((attack_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    else:
        attack_norm = np.zeros_like(attack_map)
        
    return {"mask": total_diff, "map": attack_norm}


def _read_image_gray_512(path: str) -> np.ndarray:
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise FileNotFoundError(f"Image not found or unreadable: {path}")
	if img.shape != (512, 512):
		# Resize cautiously if needed to meet challenge constraints
		img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
	return img


def _load_watermark_bits(path: str, target_len: int = 1024) -> np.ndarray:
	bits = None
	ext = os.path.splitext(path)[1].lower()
	if ext == ".npy":
		arr = np.load(path)
		bits = np.asarray(arr).astype(np.int32).flatten()
		# If values are not 0/1, threshold at median
		if not np.array_equal(np.unique(bits), [0]) and not np.array_equal(np.unique(bits), [0, 1]):
			bits = (arr.flatten() > np.median(arr)).astype(np.int32)
	else:
		# Try reading as text of 0/1
		try:
			with open(path, "r", encoding="utf-8") as f:
				s = f.read()
			nums = [c for c in s if c in ("0", "1")]
			if len(nums) > 0:
				bits = np.fromiter((1 if c == "1" else 0 for c in nums), dtype=np.int32)
		except Exception:
			pass
		if bits is None:
			# Fallback: raw bytes
			raw = np.fromfile(path, dtype=np.uint8)
			if raw.size == 0:
				raise ValueError("Watermark file is empty or unsupported format.")
			bits = (raw & 1).astype(np.int32)

	if bits.size < target_len:
		# Repeat cyclically
		reps = int(np.ceil(target_len / bits.size))
		bits = np.tile(bits, reps)
	if bits.size > target_len:
		bits = bits[:target_len]
	return bits.astype(np.uint8)


def _hanning2d(h: int, w: int) -> np.ndarray:
	wy = np.hanning(h)
	wx = np.hanning(w)
	return np.outer(wy, wx).astype(np.float32)


def _compute_logpolar_fft_template(img_u8: np.ndarray, out_size=(360, 200)) -> np.ndarray:
	# Window to reduce spectral leakage
	h, w = img_u8.shape
	win = _hanning2d(h, w)
	img = img_u8.astype(np.float32) / 255.0
	imgw = img * win
	F = np.fft.fftshift(np.fft.fft2(imgw))
	mag = np.log1p(np.abs(F)).astype(np.float32)

	# Normalize magnitude for stability
	if mag.max() > 0:
		mag = mag / (mag.max() + 1e-8)

	center = (w / 2.0, h / 2.0)
	r_max = np.hypot(center[0], center[1])
	M = out_size[0] / np.log(r_max + 1e-6)
	lp = cv2.logPolar(mag, center, M, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
	# Resize to exact target size if needed
	if lp.shape != out_size:
		lp = cv2.resize(lp, (out_size[1], out_size[0]), interpolation=cv2.INTER_AREA)
	return lp.astype(np.float32)


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
	# CSF-inspired masking: reduce strength near DC, increase towards mid frequencies
	# beta kept for backward-compat in signature, not used
	a = np.zeros((8, 8), dtype=np.float32)
	for u in range(8):
		for v in range(8):
			freq = np.sqrt(float(u * u + v * v))
			# Empirical CSF curve (lower gain at low freq, higher at mid)
			csf = np.exp(-0.25 * (freq / 4.5) ** 2)
			a[u, v] = alpha0 * (1.0 - csf) + 0.001  # small floor
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
	# Normalize to [0.8, 1.2]
	vmin, vmax = np.percentile(var_map, [5, 95])
	if vmax > vmin:
		var_norm = np.clip((var_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
	else:
		var_norm = np.zeros_like(var_map)
	return 0.8 + 0.4 * var_norm


def _pilot_tones_spatial(h: int, w: int, amp: float = 0.003) -> np.ndarray:
	# Low-frequency sinusoidal pilots
	y = np.arange(h, dtype=np.float32)[:, None]
	x = np.arange(w, dtype=np.float32)[None, :]
	# Frequencies in cycles across the image
	tones = [
		(3, 0), (0, 3), (4, 4), (5, -5),
	]
	sig = np.zeros((h, w), dtype=np.float32)
	for fy, fx in tones:
		sig += np.cos(2.0 * np.pi * (fy * y / h + fx * x / w))
	sig /= max(len(tones), 1)
	return amp * sig


def _dwt2_level2(img_f32: np.ndarray):
	if pywt is None:
		# Fallback: treat as no DWT, embed entirely in a single subband
		return {"LL2": img_f32, "LH2": None, "HL2": None, "HH2": None, "_fallback": True}
	wavelet = "bior4.4"  # approx JPEG2000 9/7
	coeffs1 = pywt.dwt2(img_f32, wavelet)
	(LL1, (LH1, HL1, HH1)) = coeffs1
	coeffs2 = pywt.dwt2(LL1, wavelet)
	(LL2, (LH2, HL2, HH2)) = coeffs2
	return {
		"LL2": LL2,
		"LH2": LH2,
		"HL2": HL2,
		"HH2": HH2,
		"L1": (LH1, HL1, HH1),
		"wavelet": wavelet,
		"_fallback": False,
	}


def _idwt2_level2(parts) -> np.ndarray:
	if pywt is None or parts.get("_fallback", False):
		return parts["LL2"].astype(np.float32)
	wavelet = parts["wavelet"]
	LL2 = parts["LL2"]
	LH2 = parts["LH2"]
	HL2 = parts["HL2"]
	HH2 = parts["HH2"]
	LH1, HL1, HH1 = parts["L1"]
	# Recompose to LL1
	LL1 = pywt.idwt2((LL2, (LH2, HL2, HH2)), wavelet)
	# Recompose to image
	img = pywt.idwt2((LL1, (LH1, HL1, HH1)), wavelet)
	return img.astype(np.float32)


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


def _block_idct(dct: np.ndarray, H: int, W: int) -> np.ndarray:
	H8, W8 = dct.shape
	img = np.zeros((H8, W8), dtype=np.float32)
	for y in range(0, H8, 8):
		for x in range(0, W8, 8):
			patch = dct[y:y + 8, x:x + 8]
			img[y:y + 8, x:x + 8] = cv2.idct(patch)
	# Pad back if needed (should match input dims)
	if H8 != H or W8 != W:
		out = np.zeros((H, W), dtype=np.float32)
		out[:H8, :W8] = img
		return out
	return img


def _ecc_encode_hamming1511(bits: np.ndarray) -> tuple[np.ndarray, dict]:
	"""Encode a bit array using Hamming(15,11) codeword mapping.
	- Input: bits array of 0/1, length L
	- Output: encoded bits, pad info
	"""
	bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
	k = 11
	n = 15
	# Pad to multiple of 11
	rem = bits.size % k
	pad = (k - rem) % k
	if pad:
		bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
	blocks = bits.reshape(-1, k)

	# Positions 1..15 (1-indexed) where parity bits are at 1,2,4,8 and data at others
	data_positions = np.array([3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32)  # 1-indexed
	parity_positions = np.array([1, 2, 4, 8], dtype=np.int32)

	# Prepare output
	out = np.zeros((blocks.shape[0], n), dtype=np.uint8)
	# Place data bits
	out[:, data_positions - 1] = blocks

	# Compute parity bits: for each parity position p, parity covers all positions with corresponding bit set
	# Using even parity
	for p in parity_positions:
		# bit index (0-based): log2(p)
		mask = (((np.arange(1, n + 1) & p) != 0).astype(np.uint8))
		# Exclude the parity position itself when summing
		mask[p - 1] = 0
		# Parity over masked positions
		parity = (np.sum(out[:, mask.astype(bool)], axis=1) % 2).astype(np.uint8)
		out[:, p - 1] = parity

	encoded = out.reshape(-1)
	meta = {"pad": int(pad), "k": k, "n": n}
	return encoded, meta


def embedding(input1, input2):
    # ... (le prime righe della funzione, fino a DWT level 2) ...
    # Read inputs
    orig_path = str(input1)
    wm_path = str(input2)

    I_orig_u8 = _read_image_gray_512(orig_path)
    I_f32 = I_orig_u8.astype(np.float32) / 255.0

    # Load watermark bits (1024)
    W_bits = _load_watermark_bits(wm_path, target_len=1024)
    orig_wm_len = int(W_bits.size)

    # Adaptive base alpha
    var_global = float(np.var(I_f32))
    if var_global < 0.015:
        alpha0 = 0.004
    elif var_global > 0.05:
        alpha0 = 0.010
    else:
        frac = (var_global - 0.015) / (0.05 - 0.015)
        alpha0 = 0.004 + (0.010 - 0.004) * frac

    # 1. CALCOLO DELLA MASCHERA DI ROBUSTEZZA A LIVELLO DI IMMAGINE
    # Questa maschera viene calcolata sull'immagine spaziale (512x512)
    robustness_data = _compute_robustness_mask(I_f32, block_size=8)
    # robustness_map ha dimensioni (64, 64) e valori [0, 1] (0 = più robusto)
    robustness_map_64 = robustness_data["map"]
    
    # Add pilot tones
    I_embed = np.clip(I_f32 + _pilot_tones_spatial(*I_f32.shape, amp=0.0025), 0.0, 1.0)

    # DWT level 2
    parts = _dwt2_level2(I_embed)
    fallback = parts.get("_fallback", False)

    # Prepare alpha and zigzag selection
    alpha_mat = _alpha_matrix(alpha0=alpha0, beta=0.6)
    zigzag = _zigzag_indices_8x8()
    
    # Seleziona le posizioni mid-frequency (ZigZag 10..30)
    sel_mask = np.zeros((8, 8), dtype=bool)
    for k, (u, v) in enumerate(zigzag):
        if 10 <= k <= 30:
            sel_mask[u, v] = True
    sel_positions = np.argwhere(sel_mask)

    # Subbands to embed into
    subbands = []
    if not fallback:
        for name in ("LH2", "HL2"):
            sb = parts[name]
            if sb is not None:
                subbands.append((name, sb))
    else:
        subbands.append(("LL2", parts["LL2"]))
        
    # Precompute per-subband variance map for masking (già presente)
    var_maps = []
    for (_, sb_arr) in subbands:
        # var_map ha dimensioni (32, 32)
        var_maps.append(_block_variance_map(sb_arr, block=8))


    # 2. RANKING DEI BLOCCHI DWT (Nuova Logica)
    
    blocks_data = [] # (sb_index, by, bx, merit)
    
    spatial_weight = 0.45 # Peso per la mascheratura (varianza)
    attack_weight = 1.0 - spatial_weight # Peso per la robustezza
    
    for si, (sb_name, sb_arr) in enumerate(subbands):
        H, W = sb_arr.shape
        H8, W8 = H // 8 * 8, W // 8 * 8
        by, bx = H8 // 8, W8 // 8
        
        # Le sottobande sono 256x256; i blocchi sono 32x32 in dimensioni di blocco 8x8.
        # La mappa di robustezza (robustness_map_64) deve essere downscalata a (32, 32)
        # per corrispondere alle dimensioni di by x bx
        
        # Downscaling approssimativo (assumiamo un fattore 2)
        r_map_sb = robustness_map_64[::2, ::2] 

        for iy in range(by):
            for ix in range(bx):
                
                # Valore Spaziale (già normalizzato a [0.8, 1.2]) - ALTO è meglio (texture)
                # Usiamo la varianza come misura spaziale/texture
                spatial_value = var_maps[si][iy, ix]

                # Valore di Attacco (Robustezza) - BASSO è meglio
                # Lo usiamo come mappa di robustezza (già normalizzata a [0, 1])
                attack_value = r_map_sb[iy, ix] 
                
                blocks_data.append({
                    'locations': (si, iy, ix), # (subband_index, by, bx)
                    'spatial_value': spatial_value,
                    'attack_value': attack_value
                })

    # A. Ranking Spaziale (Varianza): ALTO è meglio. Ordine Decrescente.
    blocks_data = sorted(blocks_data, key=lambda k: k['spatial_value'], reverse=True)
    for i in range(len(blocks_data)):
        # Il blocco con la varianza più alta (i=0) ha merito 0. Più è bassa la varianza, più è alto il merito iniziale.
        blocks_data[i]['merit'] = i * spatial_weight

    # B. Ranking Robustezza (Attacco): BASSO è meglio. Ordine Crescente.
    blocks_data = sorted(blocks_data, key=lambda k: k['attack_value'], reverse=False)
    for i in range(len(blocks_data)):
        # Il blocco più robusto (i=0) ha merito 0. Meno è robusto, più è alto il merito aggiunto.
        blocks_data[i]['merit'] += i * attack_weight

    # C. Ordinamento Finale: Il blocco con il 'merit' totale più BASSO è il MIGLIORE.
    blocks_data = sorted(blocks_data, key=lambda k: k['merit'], reverse=True)

    # Seleziona i migliori X blocchi (ad esempio, il 75% dei blocchi totali)
    n_total_blocks = len(blocks_data)
    n_best_blocks = int(n_total_blocks * 0.75) 
    
    # Seleziona i blocchi con il 'merit' più basso (pop)
    best_blocks = []
    for _ in range(n_best_blocks):
        try:
            best_blocks.append(blocks_data.pop())
        except IndexError:
            break

    # Blocchi selezionati finali in un set per la ricerca veloce
    selected_block_locations = set(b['locations'] for b in best_blocks)
    
    # 3. FILTRAGGIO DELLE POSIZIONI PER L'EMBEDDING (Nuova Logica)

    all_positions = []  # (sb_index, by, bx, u, v)
    sb_arrays = []
    sb_shapes = []
    for si, (sb_name, sb_arr) in enumerate(subbands):
        H, W = sb_arr.shape
        H8, W8 = H // 8 * 8, W // 8 * 8
        by, bx = H8 // 8, W8 // 8
        sb_arrays.append(sb_arr)
        sb_shapes.append((H, W, by, bx))
        
        for iy in range(by):
            for ix in range(bx):
                # Filtra solo i blocchi selezionati dal ranking!
                if (si, iy, ix) in selected_block_locations: 
                    for (u, v) in sel_positions:
                        all_positions.append((si, iy, ix, int(u), int(v)))


    # ... (Il resto del codice rimane invariato, utilizzando 'all_positions' filtrato) ...
    
    total_positions = len(all_positions) # Ora è ridotto!
    # ECC encode (Hamming 15,11) con padding
    payload, ecc_meta = _ecc_encode_hamming1511(W_bits.astype(np.uint8))
    # ... (tutto il resto del codice) ...
    
    # Rimosso per brevità, ma il resto della funzione continua qui!
    # ...
    
    if len(payload_perm) == 0:
        raise ValueError("Empty payload after ECC.")
    R = max(1, min(32, total_positions // len(payload_perm)))
    usable = R * len(payload_perm)
    if usable == 0:
        raise RuntimeError("No available DCT positions to embed.")

    # ... (Il resto della funzione, inclusi PRNG, Permutazione, Modifica DCT e ricostruzione) ...
    # ... (fino al salvataggio dello stato e il return finale) ...

    # ... (continua da: dct_subbands[si]...)
    
    # PRNG seed
    base_name = os.path.basename(orig_path).encode("utf-8")
    seed_bytes = sha256(base_name).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)

    # Permute payload bits
    perm_idx = rng.permutation(len(payload))
    payload_perm = payload[perm_idx]

    # Determine replication factor R to fit available coefficients
    if len(payload_perm) == 0:
        raise ValueError("Empty payload after ECC.")
    R = max(1, min(32, total_positions // len(payload_perm)))
    usable = R * len(payload_perm)
    if usable == 0:
        raise RuntimeError("No available DCT positions to embed.")

    # Randomize coefficient order and take the first `usable`
    pos_idx = rng.permutation(total_positions)[:usable]

    # Modify DCT coefficients block-wise per subband
    modified_subbands = []
    embed_positions = []

    # Prepare per-subband DCT arrays
    dct_subbands = []
    for (_, sb_arr) in subbands:
        dct_subbands.append(_block_dct(sb_arr))

    # Iterate over payload bits and assigned positions
    sgn_bits = np.where(payload_perm > 0, 1.0, -1.0).astype(np.float32)
    alpha_mat_f = alpha_mat.astype(np.float32)

    for bi in range(len(payload_perm)):
        bit_sgn = sgn_bits[bi]
        # R replications
        start = bi * R
        for ri in range(R):
            p_idx = pos_idx[start + ri]
            si, by, bx, u, v = all_positions[p_idx]

            # local masking scale from variance map (già calcolata)
            vm = var_maps[si]
            by_cl = min(by, vm.shape[0] - 1)
            bx_cl = min(bx, vm.shape[1] - 1)
            mask_scale = float(vm[by_cl, bx_cl])

            # random spread-spectrum sign per coeff
            s = 1.0 if rng.rand() > 0.5 else -1.0
            delta = alpha_mat_f[u, v] * bit_sgn * s * mask_scale

            # Apply to DCT coefficient
            dct_arr = dct_subbands[si]
            y0 = by * 8
            x0 = bx * 8
            dct_arr[y0 + u, x0 + v] += delta

            embed_positions.append((si, by, bx, u, v))

    # Inverse DCT to get modified subbands
    for si, ((_, sb_arr), dct_arr) in enumerate(zip(subbands, dct_subbands)):
        H, W = sb_arr.shape
        rec = _block_idct(dct_arr, H, W)
        modified_subbands.append(rec)

    # Put back into parts and reconstruct spatial image
    if not fallback:
        for (name, _), rec in zip(subbands, modified_subbands):
            parts[name] = rec
        I_wm = _idwt2_level2(parts)
    else:
        parts["LL2"] = modified_subbands[0]
        I_wm = parts["LL2"].astype(np.float32)

    I_wm = np.clip(I_wm, 0.0, 1.0)
    
    # ... (WPSNR adjustment, salvataggio stato, e return finale) ...
    # (continua con la parte finale del codice originale)
    
    # Compute LP_ref template on the final watermarked image (for detection phase)
    I_wm_u8 = (I_wm * 255.0 + 0.5).astype(np.uint8)
    LP_ref = _compute_logpolar_fft_template(I_wm_u8, out_size=(360, 200))

    # Build and save state
    M_embed = np.array(embed_positions, dtype=np.int16)
    ecc_params = {
        "type": "hamming1511",
        "n": int(ecc_meta["n"]),
        "k": int(ecc_meta["k"]),
        "t": 1,
        "pad": int(ecc_meta["pad"]),
        "orig_bits": orig_wm_len,
    }
    
    state = {
        "LP_ref": LP_ref.astype(np.float32),
        "M_embed": M_embed,
        "seed": int(seed),
        "ecc_params": ecc_params,
        "alpha_matrix": alpha_mat.astype(np.float32),
        "tau": float(0.88),
        "subbands": [name for (name, _) in subbands],
        "R": int(R),
        "zigzag_range": (10, 30),
        "dims": {
            "image": (512, 512),
            "subband_shapes": [(arr.shape[0], arr.shape[1]) for (_, arr) in subbands],
        },
        "target_wpsnr": 64.0,
    }

    state_path = os.path.join(_ROOT_DIR, "state.pkl")
    try:
        with open(state_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        with open(os.path.join(os.getcwd(), "state.pkl"), "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return I_wm_u8

