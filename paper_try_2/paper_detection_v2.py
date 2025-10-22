import cv2
import numpy as np
import pywt
from wpsnr import wpsnr   # la tua implementazione ufficiale

FIXED_SEED = 42
MASK = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]
WATERMARK_LEN = 1024

def _make_pn_sequences(seed=FIXED_SEED):
    rng = np.random.default_rng(seed)
    PN0 = rng.standard_normal(len(MASK)).astype(np.float32)
    PN1 = rng.standard_normal(len(MASK)).astype(np.float32)
    # normalizziamo per potenza unitaria (utile per correlazioni stabili)
    PN0 /= (np.linalg.norm(PN0) + 1e-12)
    PN1 /= (np.linalg.norm(PN1) + 1e-12)
    return PN0, PN1

def _extract_bits_from_deltas(delta_band, pn0, pn1):
    """
    delta_band: array 64x64 (una delle bande HL3, LH3, HL2, LH2)
    restituisce vettore di bit estratti (per quella banda) e conf (confidence).
    """
    bits = []
    confs = []
    for by in range(0, 64, 4):
        for bx in range(0, 64, 4):
            block = delta_band[by:by+4, bx:bx+4]
            # calcolo DCT del blocco delta (se delta è in coeff DWT, ma qui delta è direttamente DCT delta)
            C = cv2.dct(block.astype(np.float32))
            x = np.array([C[u,v] for (u,v) in MASK], dtype=np.float32)
            # correlazioni normalizzate
            den = (np.linalg.norm(x) * 1.0) + 1e-12
            rho0 = float((x @ pn0) / den)
            rho1 = float((x @ pn1) / den)
            bit = 1 if rho1 > rho0 else 0
            bits.append(bit)
            confs.append(rho1 - rho0)
    return np.array(bits, dtype=np.uint8), np.array(confs, dtype=np.float32)

def extract_bits_from_image_pair(I_ref, I_test):
    """
    Estrae i bit dalla coppia (ref, test) usando come delta C_test - C_ref.
    Ritorna vettore 1024 bit (ordine: per banda b=0..3, scansione by,bx).
    """
    # 1) DWT 3 livelli
    coeffs_ref = pywt.wavedec2(I_ref.astype(np.float32), wavelet='db2', level=3)
    coeffs_test = pywt.wavedec2(I_test.astype(np.float32), wavelet='db2', level=3)

    # estrai le 4 bande nello stesso ordine usato nell'embedding
    (LH3_r, HL3_r, HH3_r), (LH2_r, HL2_r, HH2_r), (LH1_r, HL1_r, HH1_r) = coeffs_ref[1], coeffs_ref[2], coeffs_ref[3]
    (LH3_t, HL3_t, HH3_t), (LH2_t, HL2_t, HH2_t), (LH1_t, HL1_t, HH1_t) = coeffs_test[1], coeffs_test[2], coeffs_test[3]
    bands_ref = [HL3_r, LH3_r, HL2_r, LH2_r]
    bands_test = [HL3_t, LH3_t, HL2_t, LH2_t]

    pn0, pn1 = _make_pn_sequences()

    all_bits = []
    all_confs = []
    for b in range(4):
        # calcola delta in dominio spaziale dei coefficienti della banda, poi prendi DCT per blocco
        delta_band = (bands_test[b] - bands_ref[b]).astype(np.float32)
        bits_b, confs_b = _extract_bits_from_deltas(delta_band, pn0, pn1)
        all_bits.append(bits_b)
        all_confs.append(confs_b)

    bits = np.concatenate(all_bits, axis=0)
    confs = np.concatenate(all_confs, axis=0)
    # truncate/pad a WATERMARK_LEN per sicurezza
    if bits.shape[0] > WATERMARK_LEN:
        bits = bits[:WATERMARK_LEN]
        confs = confs[:WATERMARK_LEN]
    elif bits.shape[0] < WATERMARK_LEN:
        pad = np.zeros(WATERMARK_LEN - bits.shape[0], dtype=np.uint8)
        bits = np.concatenate([bits, pad])
        confs = np.concatenate([confs, np.zeros_like(pad, dtype=np.float32)])
    return bits, confs

def detection(orig_path, watermarked_path, attacked_path, tau=0.75):
    """
    Detection non-blind.
    orig_path: path immagine originale (uint8 grayscale)
    watermarked_path: path immagine watermarkata (o immagine array)
    attacked_path: path immagine attaccata (o image array)
    tau: soglia di similarità (0..1) per dichiarare PRESENCE
    Restituisce: (present_flag (0/1), sim_bits, wpsnr_between_wm_and_att)
    """
    # read images (accept path or numpy array)
    if isinstance(orig_path, str):
        I_orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    else:
        I_orig = orig_path.copy()
    if isinstance(watermarked_path, str):
        I_w = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
    else:
        I_w = watermarked_path.copy()
    if isinstance(attacked_path, str):
        I_att = cv2.imread(attacked_path, cv2.IMREAD_GRAYSCALE)
    else:
        I_att = attacked_path.copy()

    # basic checks
    if I_orig is None or I_w is None or I_att is None:
        raise ValueError("Immagini non caricate correttamente in detection_nonblind")

    # extract bits from (orig -> watermarked) and (orig -> attacked)
    bits_w, conf_w = extract_bits_from_image_pair(I_orig, I_w)
    bits_att, conf_att = extract_bits_from_image_pair(I_orig, I_att)

    # similarity (frazione di bit uguali)
    hd = np.count_nonzero(bits_w ^ bits_att)
    sim = 1.0 - (hd / float(WATERMARK_LEN))

    # wpsnr (report)
    try:
        wpsnr_val = float(wpsnr(I_w, I_att))
    except Exception:
        wpsnr_val = -1.0

    present_flag = 1 if sim >= tau else 0
    return int(present_flag), float(wpsnr_val)
