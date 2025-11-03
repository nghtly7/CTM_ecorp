import os
import cv2
import numpy as np
import pywt

def embedding(input1, input2="ecorp.npy"):
    """
    DWT-DCT embedding con:
      - alpha adattivo per blocco (activity^gamma)
      - maschera HVS coeff-per-coeff
      - equalizzazione per blocco (1/sigma mid-band)
      - auto-tuning su kappa per WPSNR target 54.00 dB
    Firma invariata: embedding(input1, input2) -> np.uint8 (512x512)
    """
    # ---------------------------
    # Parametri "interni" (tunabili editando il file)
    # ---------------------------
    TARGET_WPSNR = 54.00
    MAX_ITERS    = 25
    WAVELET      = "db2"
    LEVELS       = 3
    FIXED_SEED   = 42
    EPS          = 1e-8

    # Pesi per banda: L3 più “forte” di L2
    BAND_WEIGHTS = {
        ("HL", 3): 1.20,
        ("LH", 3): 1.15,
        ("HL", 2): 0.70,
        ("LH", 2): 0.70,
    }

    # Maschera HVS coeff-per-coeff sui 7 mid-band (4x4 DCT)
    HVS_WEIGHTS = np.array([0.55, 0.60, 0.95, 1.10, 1.10, 1.25, 1.25], dtype=np.float32)

    # Esponente per l’attività: alpha_base ∝ (std_midband)^gamma
    ACTIVITY_GAMMA = 0.65

    # ---------------------------
    # I/O robusto
    # ---------------------------
    I = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if I is None:
        raise FileNotFoundError(f"[embedding] Impossibile leggere l'immagine: {input1}")
    I = I.astype(np.float32)

    if input2 is None or (isinstance(input2, str) and not os.path.exists(input2)):
        rngtmp = np.random.default_rng(FIXED_SEED)
        Wbits = rngtmp.integers(0, 2, size=1024, dtype=np.uint8)
        np.save("generated_watermark.npy", Wbits)
    else:
        Wbits = np.load(input2)
        if Wbits.ndim != 1:
            Wbits = Wbits.flatten()
        Wbits = Wbits.astype(np.uint8)

    if len(Wbits) != 1024:
        raise ValueError(f"[embedding] Il watermark deve avere 1024 bit, trovato {len(Wbits)}.")

    # ---------------------------
    # DWT (3 livelli)
    # coeffs = (LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1))
    # ---------------------------
    coeffs = pywt.wavedec2(I, wavelet=WAVELET, level=LEVELS)
    (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs[1], coeffs[2], coeffs[3]

    bands = [
        ("HL", 3, HL3),
        ("LH", 3, LH3),
        ("HL", 2, HL2),
        ("LH", 2, LH2),
    ]

    # ---------------------------
    # Mid-band 4x4 + PN ortonormalizzate
    # ---------------------------
    MASK_IDX = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]
    rng = np.random.default_rng(FIXED_SEED)
    PN0 = rng.standard_normal(len(MASK_IDX)) * 1.5
    PN1 = rng.standard_normal(len(MASK_IDX)) * 1.5  
    # Ortonormalizzazione (migliora distinzione 0/1)
    PN0 = PN0 / (np.linalg.norm(PN0) + EPS)
    PN1 = PN1 - (PN0 @ PN1) * PN0
    PN1 = PN1 / (np.linalg.norm(PN1) + EPS)

    # ---------------------------
    # WPSNR wrapper (usa la tua implementazione)
    # ---------------------------
    def _wpsnr_wrapper(Au8, Bu8) -> float:
        try:
            from wpsnr import wpsnr as _wpsnr
        except Exception:
            try:
                from wpsnr import wpsnr as _wpsnr
            except Exception:
                raise ImportError("[embedding] Impossibile importare wpsnr() dalla detection.")
        return float(_wpsnr(Au8, Bu8))

    # ---------------------------
    # Embedding con kappa (una passata)
    # ---------------------------
    def _embed_with_kappa(kappa: float) -> np.ndarray:
        HL3c, LH3c, HL2c, LH2c = [b[2].copy() for b in bands]
        band_map = {("HL",3): HL3c, ("LH",3): LH3c, ("HL",2): HL2c, ("LH",2): LH2c}

        idx = 0
        for (name, level, B) in [("HL",3,HL3c), ("LH",3,LH3c), ("HL",2,HL2c), ("LH",2,LH2c)]:
            r_band = BAND_WEIGHTS[(name, level)]
            for by in range(0, 64, 4):
                for bx in range(0, 64, 4):
                    block = B[by:by+4, bx:bx+4].astype(np.float32, copy=True)
                    C = cv2.dct(block)

                    # Activity: std dei 7 mid-band del blocco
                    mb = np.array([C[u,v] for (u,v) in MASK_IDX], dtype=np.float32)
                    mb_std = float(np.std(mb))
                    
                    # 🔸 SKIP opzionale per blocchi troppo piatti
                    if mb_std < 1.5:      # ← puoi regolare la soglia 1.5 in base al tipo d’immagine
                        idx += 1
                        continue
                    
                    # α di base (activity^gamma) con peso banda
                    alpha_base = kappa * r_band * (mb_std + EPS)**ACTIVITY_GAMMA
                    # Equalizzazione per blocco: rende l'effetto più uniforme
                    eq = 1.0 / (mb_std + EPS)

                    bit = int(Wbits[idx])
                    PN  = PN1 if bit == 1 else PN0

                    # Aggiornamento coeff-per-coeff con HVS + equalizzazione
                    for k, (u, v) in enumerate(MASK_IDX):
                        C[u, v] += alpha_base * HVS_WEIGHTS[k] * PN[k] * eq

                    B[by:by+4, bx:bx+4] = cv2.idct(C)
                    idx += 1

            band_map[(name, level)] = B

        # Ricomposizione e IDWT
        new_coeffs = list(coeffs)
        new_coeffs[1] = (band_map[("LH",3)], band_map[("HL",3)], HH3)
        new_coeffs[2] = (band_map[("LH",2)], band_map[("HL",2)], HH2)
        new_coeffs[3] = (LH1, HL1, HH1)
        Iw_f = pywt.waverec2(new_coeffs, wavelet=WAVELET)
        return np.clip(Iw_f, 0, 255).astype(np.float32)

    # ---------------------------
    # Auto-tuning di kappa (raddoppio + bisection)
    # ---------------------------
    k_lo, k_hi = 0.0, 1.0
    Iw_hi = _embed_with_kappa(k_hi)
    W_hi = _wpsnr_wrapper(I.astype(np.uint8), Iw_hi.astype(np.uint8))
    tries = 0
    while W_hi >= TARGET_WPSNR and tries < 12:
        k_hi *= 2.0
        Iw_hi = _embed_with_kappa(k_hi)
        W_hi = _wpsnr_wrapper(I.astype(np.uint8), Iw_hi.astype(np.uint8))
        tries += 1

    best_Iw = None
    for _ in range(MAX_ITERS):
        k_mid = 0.5*(k_lo + k_hi)
        Iw_mid = _embed_with_kappa(k_mid)
        W_mid  = _wpsnr_wrapper(I.astype(np.uint8), Iw_mid.astype(np.uint8))

        if W_mid >= TARGET_WPSNR:
            best_Iw = Iw_mid
            k_lo    = k_mid
        else:
            k_hi = k_mid

        if abs(k_hi - k_lo) < 1e-3:
            break

    if best_Iw is None:
        best_Iw = Iw_hi if W_hi >= TARGET_WPSNR else _embed_with_kappa(k_lo)

    return np.clip(best_Iw, 0, 255).astype(np.uint8)


# ================================================================
# 🎛️  PARAMETRI AVANZATI DI TUNING DELL'EMBEDDING
# ================================================================
# Questi parametri controllano *dove* e *come* viene distribuita
# l'energia del watermark all’interno delle bande DWT/DCT.
# L’auto-tuning di kappa garantirà comunque che il WPSNR finale
# resti ≥ TARGET_WPSNR (54 dB), quindi puoi spingere la robustezza
# senza rischiare di scendere sotto la soglia percettiva.
#
# 📌 PARAMETRI CHIAVE
# ------------------------------------------------
# 1) BAND_WEIGHTS → bilancia la potenza fra bande
#    • HL3/LH3 (livello 3)  → frequenze più basse, più robuste
#    • HL2/LH2 (livello 2)  → frequenze più alte, più sensibili
#    → Aumentare HL3/LH3 migliora la robustezza JPEG e resize.
#      Range consigliato:
#         L3 = 1.0–1.5
#         L2 = 0.6–0.9
#    Esempio “robusto”:
#         ("HL",3): 1.35, ("LH",3): 1.30,
#         ("HL",2): 0.70, ("LH",2): 0.70
#
# 2) HVS_WEIGHTS → pesi percettivi sui 7 coeff. mid-band DCT
#    • I primi due valori (0.55–0.60) corrispondono a freq. basse,
#      mantenere < 0.7 per evitare artefatti visivi.
#    • Gli ultimi valori (1.15–1.35) sono le freq. più alte e possono
#      essere aumentati per migliorare la robustezza a compressione.
#    Esempio “robusto JPEG”:
#         [0.55, 0.60, 0.95, 1.10, 1.10, 1.30, 1.35]
#
# 3) ACTIVITY_GAMMA → controlla l’adattività locale
#    α_base ∝ (std_midband)^γ
#    • Maggiore γ (0.7–0.8) → watermark più forte nelle zone ricche
#      di texture, invisibile nelle zone piatte.
#    • Minore γ (0.45–0.55) → distribuzione più uniforme (immagini lisce).
#    Tipicamente: 0.60–0.75 bilancia bene robustezza e invisibilità.
#
# 4) SKIP DI BLOCCHI PIATTI (facoltativo)
#    Se vuoi evitare embedding in blocchi troppo omogenei:
#        if mb_std < 1.8: continue
#    • Concentra l’energia nei blocchi “sicuri”.
#      Range utile: 1.5–2.0.
#
# Tutti questi parametri vengono poi scalati automaticamente da kappa
# per rispettare il TARGET_WPSNR → puoi testarli liberamente.
#
# PROFILI ESEMPLIFICATIVI
# ------------------------------------------------
# • Bilanciato (default robusto):
#       BAND_WEIGHTS = {("HL",3):1.20,("LH",3):1.15,("HL",2):0.70,("LH",2):0.70}
#       HVS_WEIGHTS  = [0.55,0.60,0.95,1.10,1.10,1.25,1.25]
#       ACTIVITY_GAMMA = 0.65
#
# • Profilo JPEG/resize aggressivo:
#       BAND_WEIGHTS = {("HL",3):1.35,("LH",3):1.30,("HL",2):0.70,("LH",2):0.70}
#       HVS_WEIGHTS  = [0.55,0.60,0.95,1.10,1.15,1.30,1.35]
#       ACTIVITY_GAMMA = 0.75–0.80
#       mb_std skip ≈ 1.8
#
# • Profilo rumore/blur distribuito:
#       BAND_WEIGHTS = {("HL",3):1.20,("LH",3):1.15,("HL",2):0.80,("LH",2):0.80}
#       HVS_WEIGHTS  = [0.55,0.60,0.95,1.10,1.15,1.25,1.30]
#       ACTIVITY_GAMMA = 0.60–0.70
# ================================================================
