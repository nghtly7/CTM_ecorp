import cv2
import numpy as np
import pywt

def embedding(input1, input2="../ecorp.npy"):
    """
    DWT-DCT embedding con alpha adattivo + maschera HVS + auto-tuning su kappa.
    - input1: path immagine originale (512x512, grayscale)
    - input2: path watermark bits (npy, vettore 1024 di {0,1})
    - return: immagine watermarked (np.uint8)
    """
    # ---------------------------
    # Parametri principali
    # ---------------------------
    TARGET_WPSNR = 54.0     # obiettivo qualità
    MAX_ITERS    = 10       # iterazioni bisection su kappa
    FIXED_SEED   = 42       # deve restare coerente con la detection
    EPS          = 1e-6

    # Pesi per banda (favorisci L3 rispetto a L2)
    BAND_WEIGHTS = {
        ("HL", 3): 1.00,
        ("LH", 3): 1.00,
        ("HL", 2): 0.70,
        ("LH", 2): 0.70,
    }

    # Maschera HVS coeff-per-coeff sui 7 mid-band della DCT 4x4
    # Ordine coerente con la tua maschera corrente (vedi sotto MASK_IDX)
    HVS_WEIGHTS = np.array([0.6, 0.6, 0.9, 1.0, 1.0, 1.1, 1.1], dtype=np.float32)

    # ---------------------------
    # I/O
    # ---------------------------
    I = cv2.imread(input1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if input2 is None:
        rngtmp = np.random.default_rng(FIXED_SEED)
        Wbits = rngtmp.integers(0, 2, size=1024, dtype=np.uint8)
        np.save("generated_watermark.npy", Wbits)
    else:
        Wbits = np.load(input2).astype(np.uint8)

    # ---------------------------
    # DWT 3 livelli
    # coeffs = (LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1))
    # ---------------------------
    coeffs = pywt.wavedec2(I, wavelet='db2', level=3)
    (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs[1], coeffs[2], coeffs[3]

    # Selezione 4 bande (ordine coerente embedding/detection)
    bands = [
        ("HL", 3, HL3),
        ("LH", 3, LH3),
        ("HL", 2, HL2),
        ("LH", 2, LH2),
    ]

    # ---------------------------
    # Mid-band DCT 4x4 (7 coeff) e PN
    # ---------------------------
    MASK_IDX = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]  # invariato rispetto al tuo codice
    rng = np.random.default_rng(FIXED_SEED)                 # fisso per coerenza con detection
    PN0 = rng.standard_normal(len(MASK_IDX)).astype(np.float32)
    PN1 = rng.standard_normal(len(MASK_IDX)).astype(np.float32)
    # Ortonormalizza PN per migliorare separazione (stesso FIXED_SEED lato detection)
    PN0 = PN0 / (np.linalg.norm(PN0) + EPS)
    PN1 = PN1 - (PN0 @ PN1) * PN0
    PN1 = PN1 / (np.linalg.norm(PN1) + EPS)

    # ---------------------------
    # Funzioni helpers
    # ---------------------------
    def _embed_with_kappa(kappa: float) -> np.ndarray:
        """
        Applica l'embedding con kappa globale: alpha_block = kappa * r_band * sqrt(activity) * HVS_coeff
        Torna l'immagine watermarked (float32 [0..255]).
        """
        # Copie locali dei subband (evita side-effects)
        HL3c, LH3c, HL2c, LH2c = [b[2].copy() for b in bands]
        band_map = {("HL",3): HL3c, ("LH",3): LH3c, ("HL",2): HL2c, ("LH",2): LH2c}

        idx = 0
        for (name, level, B) in [("HL",3,HL3c),("LH",3,LH3c),("HL",2,HL2c),("LH",2,LH2c)]:
            r_band = BAND_WEIGHTS[(name, level)]
            # Scorri blocchi 4x4
            for by in range(0, 64, 4):
                for bx in range(0, 64, 4):
                    block = B[by:by+4, bx:bx+4].astype(np.float32, copy=True)
                    C = cv2.dct(block)

                    # Activity = std dei 7 mid-band del blocco (robusta e veloce)
                    mb = np.array([C[u,v] for (u,v) in MASK_IDX], dtype=np.float32)
                    activity = float(np.std(mb))  # >=0
                    # α_block per coeff = kappa * r_band * sqrt(activity+eps) * HVS_weight[k]
                    alpha_base = kappa * r_band * np.sqrt(activity + 1e-8)

                    # Bit e PN locali
                    bit = int(Wbits[idx])
                    PN = PN1 if bit == 1 else PN0

                    # Embedding coeff-per-coeff con HVS
                    for k, (u, v) in enumerate(MASK_IDX):
                        C[u, v] += alpha_base * HVS_WEIGHTS[k] * PN[k]

                    # IDCT nel subband
                    B[by:by+4, bx:bx+4] = cv2.idct(C)
                    idx += 1

            # scrivi back nel contenitore
            band_map[(name, level)] = B

        # Ricomponi coeffs e fai IDWT
        new_coeffs = list(coeffs)
        new_coeffs[1] = (band_map[("LH",3)], band_map[("HL",3)], HH3)
        new_coeffs[2] = (band_map[("LH",2)], band_map[("HL",2)], HH2)
        new_coeffs[3] = (LH1, HL1, HH1)
        Iw_f = pywt.waverec2(new_coeffs, wavelet='db2')
        return np.clip(Iw_f, 0, 255).astype(np.float32)

    def _wpsnr_wrapper(Au8: np.ndarray, Bu8: np.ndarray) -> float:
        # Usa la tua WPSNR dal detection (CSF) per coerenza challenge
        try:
            # Se i moduli sono nello stesso folder
            from paper_detection_v1 import wpsnr as _wpsnr
        except Exception:
            try:
                # Se hai una package path (adatta se serve)
                from paper_try_2.paper_detection_v1 import wpsnr as _wpsnr
            except Exception:
                raise ImportError("Impossibile importare wpsnr() dalla detection. Assicurati che sia nel PYTHONPATH.")
        return float(_wpsnr(Au8, Bu8))  # la tua implementazione gestisce normalizzazioni

    # ---------------------------
    # Auto-tuning su kappa (bisection)
    # ---------------------------
    # 1) trova un upper bound tale che WPSNR < target (se non lo trovi, aumenta geometricamente)
    k_lo, k_hi = 0.0, 1.0
    Iw_hi = _embed_with_kappa(k_hi)
    W_hi = _wpsnr_wrapper(I.astype(np.uint8), Iw_hi.astype(np.uint8))
    tries = 0
    while W_hi >= TARGET_WPSNR and tries < 12:
        k_hi *= 2.0
        Iw_hi = _embed_with_kappa(k_hi)
        W_hi = _wpsnr_wrapper(I.astype(np.uint8), Iw_hi.astype(np.uint8))
        tries += 1
    # Se anche con kappa molto grande non scendi sotto target, usa k_hi corrente

    # 2) bisection per centrare TARGET_WPSNR (tenendosi sul lato >= target)
    best_Iw = None
    best_k  = k_lo
    for _ in range(MAX_ITERS):
        k_mid = 0.5*(k_lo + k_hi)
        Iw_mid = _embed_with_kappa(k_mid)
        W_mid  = _wpsnr_wrapper(I.astype(np.uint8), Iw_mid.astype(np.uint8))

        if W_mid >= TARGET_WPSNR:
            # possiamo alzare
            best_Iw = Iw_mid
            best_k  = k_mid
            k_lo    = k_mid
        else:
            # troppo forte → abbassa
            k_hi = k_mid

        if abs(k_hi - k_lo) < 1e-3:
            break

    # Fallback se per qualche motivo best_Iw non è stato settato
    if best_Iw is None:
        best_Iw = Iw_hi if W_hi >= TARGET_WPSNR else _embed_with_kappa(k_lo)

    return np.clip(best_Iw, 0, 255).astype(np.uint8)
