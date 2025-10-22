import cv2
import numpy as np
import pywt
from wpsnr import wpsnr as WPSNR   # usa la funzione ufficiale che hai

# Parametri che devono corrispondere all'embedding
FIXED_SEED = 42
MASK = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]
WATERMARK_LEN = 1024
TAU_DEFAULT = 0.6

# Prepara PN normalizzate (stesso seed dell'embedding)
def _make_pn_sequences(seed=FIXED_SEED):
    rng = np.random.default_rng(seed)
    PN0 = rng.standard_normal(len(MASK)).astype(np.float32)
    PN1 = rng.standard_normal(len(MASK)).astype(np.float32)
    PN0 /= (np.linalg.norm(PN0) + 1e-12)
    PN1 /= (np.linalg.norm(PN1) + 1e-12)
    return PN0, PN1

PN0, PN1 = _make_pn_sequences()

def _extract_bits_from_pair(I_ref, I_test):
    """
    Estrae i WATERMARK_LEN bit da coppia (original, test) usando Δ = DCT(band_test) - DCT(band_ref)
    Ritorna: bits (np.uint8, len=WATERMARK_LEN), confs (float per bit: rho1-rho0)
    """
    # assicuriamoci float
    I_ref = I_ref.astype(np.float32)
    I_test = I_test.astype(np.float32)

    # DWT 3 livelli (db2) su entrambe
    coeffs_ref  = pywt.wavedec2(I_ref, 'db2', level=3)
    coeffs_test = pywt.wavedec2(I_test, 'db2', level=3)

    # nella struttura pywt: coeffs = (LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1))
    (LH3_r, HL3_r, HH3_r), (LH2_r, HL2_r, HH2_r), (LH1_r, HL1_r, HH1_r) = coeffs_ref[1], coeffs_ref[2], coeffs_ref[3]
    (LH3_t, HL3_t, HH3_t), (LH2_t, HL2_t, HH2_t), (LH1_t, HL1_t, HH1_t) = coeffs_test[1], coeffs_test[2], coeffs_test[3]

    # selezione bande nello stesso ordine dell'embedding
    bands_ref  = [HL3_r, LH3_r, HL2_r, LH2_r]
    bands_test = [HL3_t, LH3_t, HL2_t, LH2_t]

    bits_list = []
    confs_list = []
    idx = 0
    for b in range(4):
        Br = bands_ref[b]
        Bt = bands_test[b]
        # assumiamo che Br e Bt abbiano dimensione 64x64 (per immagini 512x512)
        for by in range(0, 64, 4):
            for bx in range(0, 64, 4):
                # DCT su blocchi 4x4 dei coefficienti della banda
                Cr = cv2.dct(Br[by:by+4, bx:bx+4].astype(np.float32))
                Ct = cv2.dct(Bt[by:by+4, bx:bx+4].astype(np.float32))

                # vettore differenza sugli indici della maschera
                dC = np.array([Ct[u,v] - Cr[u,v] for (u,v) in MASK], dtype=np.float32)

                den = (np.linalg.norm(dC) + 1e-12)
                rho0 = float((dC @ PN0) / den)
                rho1 = float((dC @ PN1) / den)

                bit = 1 if rho1 > rho0 else 0
                conf = rho1 - rho0

                bits_list.append(bit)
                confs_list.append(conf)
                idx += 1

    bits = np.array(bits_list, dtype=np.uint8)
    confs = np.array(confs_list, dtype=np.float32)

    # assicurati lunghezza WATERMARK_LEN (tronca o pad con zeri se necessario)
    if bits.shape[0] > WATERMARK_LEN:
        bits = bits[:WATERMARK_LEN]
        confs = confs[:WATERMARK_LEN]
    elif bits.shape[0] < WATERMARK_LEN:
        pad_len = WATERMARK_LEN - bits.shape[0]
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
        confs = np.concatenate([confs, np.zeros(pad_len, dtype=np.float32)])

    return bits, confs

def detection(orig, wm, att, tau=TAU_DEFAULT):
    """
    Detection non-blind.
    orig, wm, att can be file paths (str) or numpy arrays (uint8 grayscale)
    Returns: (present_flag (0/1), similarity (0..1), wpsnr_between_wm_and_att)
    """
    # carica immagini se passati path
    if isinstance(orig, str):
        I_orig = cv2.imread(orig, cv2.IMREAD_GRAYSCALE)
    else:
        I_orig = orig.copy()
    if isinstance(wm, str):
        I_w = cv2.imread(wm, cv2.IMREAD_GRAYSCALE)
    else:
        I_w = wm.copy()
    if isinstance(att, str):
        I_att = cv2.imread(att, cv2.IMREAD_GRAYSCALE)
    else:
        I_att = att.copy()

    # controlli
    if I_orig is None or I_w is None or I_att is None:
        raise ValueError("detection_nonblind: immagine non caricata correttamente")

    # Estrai bit da (orig -> wm) e (orig -> att)
    bits_w, conf_w = _extract_bits_from_pair(I_orig, I_w)
    bits_att, conf_att = _extract_bits_from_pair(I_orig, I_att)

    # Similarità bit (1 - normalized Hamming)
    hd = int(np.count_nonzero(bits_w ^ bits_att))
    sim = 1.0 - (hd / float(WATERMARK_LEN))

    present_flag = 1 if sim >= tau else 0

    # WPSNR (report) tra watermarked e attacked (è la metrica richiesta)
    try:
        wpsnr_val = float(WPSNR(I_orig, I_att))
    except Exception:
        wpsnr_val = -1.0

    return int(present_flag), float(wpsnr_val)
