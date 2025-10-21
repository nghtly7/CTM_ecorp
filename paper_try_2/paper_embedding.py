import cv2
import numpy as np
import pywt
# Parametri embedding
alpha = 3.0      # embedding strength (aumentato per robustezza)
FIXED_SEED = 42
# WATERMARK_SIZE = 1024
# np.random.seed(FIXED_SEED)


def embedding(input1, input2=None):
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
    mask = [(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2)]
    rng = np.random.default_rng(FIXED_SEED)      # fisso per coerenza detection
    PN0 = rng.standard_normal(len(mask))
    PN1 = rng.standard_normal(len(mask))

    # 5) embed 1 bit per blocco, 256 blocchi per banda × 4 bande = 1024
    idx = 0
    for b in range(4):
        B = bands[b]
        for by in range(0, 64, 4):
            for bx in range(0, 64, 4):
                block = B[by:by+4, bx:bx+4].copy()
                C = cv2.dct(block)
                bit = int(Wbits[idx])
                PN = PN1 if bit==1 else PN0
                for k,(u,v) in enumerate(mask):
                    C[u,v] += alpha * PN[k]
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
