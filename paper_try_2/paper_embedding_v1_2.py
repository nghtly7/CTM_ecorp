import cv2
import numpy as np
import pywt


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


def embedding(input1, input2='../ecorp.npy'):
    
    # Parametri embedding
    alpha = 3.0     # embedding strength (aumentato per robustezza)
    FIXED_SEED = 42
    #WATERMARK_SIZE = 1024
    #np.random.seed(FIXED_SEED)

    # 1) I/O
    I = cv2.imread(input1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # Se watermark non passato, lo genero
    if input2 is None:
        rng = np.random.default_rng(FIXED_SEED)
        Wbits = rng.integers(0, 2, size=1024, dtype=np.uint8)  # vettore 0/1
        np.save("generated_watermark.npy", Wbits)  # lo salvo per la detection
    else:
        Wbits = np.load(input2).astype(np.uint8)    

    # 2) DWT 3 livelli
    coeffs = pywt.wavedec2(I, wavelet='db2', level=3)   # coeffs = (LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1))

    # 3) scegli 4 sottobande: HL13, LH13, HL23, LH23  (seguendo lo schema del paper)
    #    NB: in pywt l’ordine in ciascun livello è (LH, HL, HH).
    (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs[1], coeffs[2], coeffs[3]
    bands = [HL3, LH3, HL2, LH2]   # adatta all’esatta corrispondenza con le 4 mappe da 64x64

    # 4) set mid-band e PN sequences
    mask = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)] # 7 mid-frequency DCT coefficients
    rng = np.random.default_rng(FIXED_SEED)      # fisso per coerenza detection
    PN0 = rng.standard_normal(len(mask)).astype(np.float32)
    PN1 = rng.standard_normal(len(mask)).astype(np.float32)
    PN0 /= (np.linalg.norm(PN0) + 1e-12)
    PN1 /= (np.linalg.norm(PN1) + 1e-12)

    # --- PRE-CALCOLO ENERGIE GLOBALI ---
    energies = []
    for b in range(4):
        B = bands[b]
        for by in range(0,64,4):
            for bx in range(0,64,4):
                block = B[by:by+4, bx:bx+4]
                C = cv2.dct(block.astype(np.float32))
                vals = np.array([C[u,v] for (u,v) in mask], dtype=np.float32)
                energies.append(np.linalg.norm(vals))

    energies = np.array(energies, dtype=np.float32)
    E_mean = float(np.mean(energies))
    E_ptp = float(np.ptp(energies) + 1e-12)   # con NumPy 2.0 si usa np.ptp()
    # --- DIMENSIONI IMMAGINE E ATTACK MAP ---
    H, W = I.shape
    attack_map = compute_attack_map(I.astype(np.uint8))

    

    # --- 5) DRY-RUN: STIMA MSE GLOBALE ---
    alpha_blocks = []
    idx = 0
    for b in range(4):
        B = bands[b]
        for by in range(0, 64, 4):
            for bx in range(0, 64, 4):
                block = B[by:by+4, bx:bx+4].copy()
                C = cv2.dct(block)

                vals = np.array([C[u,v] for (u,v) in mask], dtype=np.float32)
                E = np.linalg.norm(vals)
                E_norm = (E - E_mean) / (E_ptp + 1e-12)
                local_mean = block.mean()
                lum_mask = 1.0 + 0.15 * (local_mean/128.0 - 1.0)

                beta = 0.6
                alpha_block = alpha * (1.0 + beta * E_norm) * lum_mask
                alpha_block = float(np.clip(alpha_block, 0.1*alpha, 2.0*alpha))

                alpha_blocks.append(alpha_block)
                idx += 1

    # --- 6) WPSNR BUDGETING ---
    pred_MSE = np.sum(np.array(alpha_blocks, dtype=np.float32)**2) / (I.shape[0] * I.shape[1])
    target_wpsnr_db = 66.0
    target_mse = (255.0**2) / (10**(target_wpsnr_db/10))
    scale = np.sqrt(target_mse / (pred_MSE + 1e-12))
    alpha *= scale   # alpha definitivo e ottimale


    # --- 7) EMBEDDING REALE (con alpha scalato) ---
    idx = 0
    beta   = 0.6
    gamma  = 0.4
    soft_t = 0.15   # soglia blocchi piatti (15% sotto l'energia media)
    soft_k = 0.6
    Q = 0.6  # quantizzazione correttiva
    for b in range(4):
        B = bands[b]
        for by in range(0, 64, 4):
            for bx in range(0, 64, 4):
                block = B[by:by+4, bx:bx+4].copy()
                C = cv2.dct(block)

                bit = int(Wbits[idx])
                # vettore host e PN
                x = np.array([C[u,v] for (u,v) in mask], dtype=np.float32)
                p = PN1 if bit==1 else PN0

                # HIR (proietta PN ortogonale a x)
                den = (x @ x) + 1e-12
                p_ortho = p - ((p @ x) / den) * x
                p_use = p if np.linalg.norm(p_ortho) < 1e-6 else p_ortho / (np.linalg.norm(p_ortho) + 1e-12)

                # energia locale + attack_score
                vals = x
                E = np.linalg.norm(vals)
                E_norm = (E - E_mean) / E_ptp
                local_mean = block.mean()
                lum_mask = 1.0 + 0.15 * (local_mean/128.0 - 1.0)

                img_y = int((by/64.0) * H)
                img_x = int((bx/64.0) * W)
                h = max(1, H//64); w = max(1, W//64)
                y0 = max(0, img_y - h//2); y1 = min(H, y0 + h)
                x0 = max(0, img_x - w//2); x1 = min(W, x0 + w)
                attack_score = float(attack_map[y0:y1, x0:x1].mean()) if (y1>y0 and x1>x0) else 0.0

                #* da capire meglio questo blocco
                alpha_block = alpha * (1.0 + beta * E_norm) * (1.0 - gamma * attack_score) * lum_mask
                if E < soft_t * E_mean:
                    #alpha_block *= soft_k #*specialmente questa parte
                    continue  # skip totale per blocchi piatti
                alpha_block = float(np.clip(alpha_block, 0.1*alpha, 2.0*alpha))
                
                 
                # # soft–scaling per blocchi piatti (perceptual skip / attenuation)
                # if E < 0.10 * E_mean:          # soglia (consigliato 0.25 – 0.35)   # attualmente soft-skip, per saltare i blocchi bianchi
                #      continue #     alpha_block *= 0.25        # attenua (oppure 0.0 se vuoi skip totale)
                # #wpsnr diminuisce leggermente, ma robustezza migliora, dipende dall'immagine

                # iniezione watermark
                for k,(u,v) in enumerate(mask):
                    C[u,v] += alpha_block * p_use[k]

                # quantizzazione correttiva
                for (u,v) in mask:
                    C[u,v] = Q * np.round(C[u,v] / Q)

                B[by:by+4, bx:bx+4] = cv2.idct(C)
                idx += 1
        bands[b] = B


    # 6) rimetti le bande modificate dentro coeffs e IDWT
    #    (ricostruisci la stessa struttura coeffs con le bande aggiornate)
    new_coeffs = list(coeffs)
    LH3n, HL3n = bands[1], bands[0]
    LH2n, HL2n = bands[3], bands[2]
    new_coeffs[1] = (LH3n, HL3n, HH3)
    new_coeffs[2] = (LH2n, HL2n, HH2)
    new_coeffs[3] = (LH1,   HL1,   HH1)
    Iw = pywt.waverec2(new_coeffs, wavelet='db2')

    Iw = np.clip(Iw, 0, 255).astype(np.uint8)
    return Iw
