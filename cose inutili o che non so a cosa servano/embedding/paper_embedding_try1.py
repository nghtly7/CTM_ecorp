import numpy as np
import pywt
from scipy.fft import dctn, idctn

# --- PARAMETRI GLOBALI ---
IMAGE_SIZE = 512
WATERMARK_SIZE = 32  # Watermark binario 32x32 = 1024 bit
ALPHA = 0.05       # Fattore di guadagno (embedding strength)
ARNOLD_ITER = 3    # Numero di iterazioni per Arnold Cat Map (chiave segreta)
PN_SEED = 42       # Seed per la generazione delle sequenze PN (chiave segreta)
BLOCK_SIZE = 4     # Dimensione del blocco DCT (4x4)

# --- FUNZIONI DI BASE ---

def arnold_cat_map(img, iterations):
    """Implementazione concettuale della Arnold Cat Map per scramble/unscramble."""
    if iterations > 0:
        return np.roll(img, iterations, axis=(0, 1))
    else:
        return np.roll(img, -iterations, axis=(0, 1)) 

def generate_pn_sequence(length, seed):
    """Genera una sequenza PN pseudo-casuale."""
    np.random.seed(seed)
    return np.random.normal(0, 1, length)

def embed_dct_block(dct_block, watermark_bit, pn0_block, pn1_block, alpha):
    """Modifica i coefficienti DCT di banda media (Equazione 5).
    
    Riceve solo la porzione PN di 8 elementi necessaria per il blocco.
    """
    num_pn_coeffs = 8 # Numero di coefficienti di banda media usati
    
    # Ottenere gli indici dei coefficienti di banda media
    mid_band_indices = list(zip(*np.unravel_index(np.arange(2, 2 + num_pn_coeffs), (BLOCK_SIZE, BLOCK_SIZE))))
    
    current_pn = pn0_block if watermark_bit == 0 else pn1_block
    
    # Inserimento nei coefficienti di banda media
    modified_block = dct_block.copy()
    
    # Modifica (X' = X + alpha * PN)
    for i, (r, c) in enumerate(mid_band_indices):
        # Assumiamo che current_pn sia di lunghezza 8
        modified_block[r, c] += alpha * current_pn[i]
        
    return modified_block

# --- FUNZIONE DI EMBEDDING (INSERIMENTO) REVISIONATA ---

def embedding_algorithm(host_image, watermark_bits, alpha, pn0_full, pn1_full, arnold_iter):
    """Implementa la procedura di embedding DWT-DCT a 3 livelli su 4 bande (1024 bit)."""
    
    # 1. DWT a 3 livelli e selezione delle 4 bande
    coeffs1 = pywt.wavedec2(host_image, 'haar', level=1)
    LL1, (HL1, LH1, HH1) = coeffs1
    
    coeffs_hl2 = pywt.wavedec2(HL1, 'haar', level=1)
    LL_HL2, (HL12_orig, LH12_orig, HH12_orig) = coeffs_hl2
    coeffs_lh2 = pywt.wavedec2(LH1, 'haar', level=1)
    LL_LH2, (HL22_orig, LH22_orig, HH22_orig) = coeffs_lh2 

    # Livello 3 (otteniamo i coefficienti di terzo livello)
    LL_HL12_3, (HL13, LH13, HH13) = pywt.wavedec2(HL12_orig, 'haar', level=1)
    LL_LH12_3, (HL23_T, LH13_T, HH23_T) = pywt.wavedec2(LH12_orig, 'haar', level=1) 
    LL_HL22_3, (HL23, LH23_T, HH23_T_2) = pywt.wavedec2(HL22_orig, 'haar', level=1) 
    LL_LH22_3, (HL33_T, LH23, HH33_T) = pywt.wavedec2(LH22_orig, 'haar', level=1)

    # Copiamo le 4 bande target per la modifica
    HL13_wm = HL13.copy()
    LH13_wm = LH13_T.copy()
    HL23_wm = HL23.copy()
    LH23_wm = LH23.copy()
    
    bands_to_modify = [HL13_wm, LH13_wm, HL23_wm, LH23_wm]
    num_pn_coeffs = 8 # Lunghezza PN per blocco

    watermark_idx = 0 
    
    # 2. Inserimento dei 1024 bit
    for target_band in bands_to_modify:
        h, w = target_band.shape
        num_blocks_h = h // BLOCK_SIZE
        num_blocks_w = w // BLOCK_SIZE
        
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                if watermark_idx >= len(watermark_bits):
                    break
                
                block = target_band[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
                dct_block = dctn(block, type=2, norm='ortho')
                
                bit_to_embed = watermark_bits[watermark_idx]
                
                # Truncamento della sequenza PN alla lunghezza necessaria per il blocco (8 elementi)
                pn0_block = pn0_full[watermark_idx * num_pn_coeffs : (watermark_idx + 1) * num_pn_coeffs]
                pn1_block = pn1_full[watermark_idx * num_pn_coeffs : (watermark_idx + 1) * num_pn_coeffs]
                
                modified_dct_block = embed_dct_block(modified_dct_block, bit_to_embed, 
                                                     pn0_block, pn1_block, alpha)
                
                modified_block = idctn(modified_dct_block, type=2, norm='ortho')
                target_band[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] = modified_block
                
                watermark_idx += 1
        if watermark_idx >= len(watermark_bits):
             break


    # 3. Ricostruzione IDWT inversa
    
    # 3. Livello IDWT
    HL12_wm = pywt.waverec2((LL_HL12_3, (HL13_wm, LH13, HH13)), 'haar')
    LH12_wm = pywt.waverec2((LL_LH12_3, (HL23_T, LH13_wm, HH23_T)), 'haar')
    HL22_wm = pywt.waverec2((LL_HL22_3, (HL23_wm, LH23_T, HH23_T_2)), 'haar')
    LH22_wm = pywt.waverec2((LL_LH22_3, (HL33_T, LH23_wm, HH33_T)), 'haar')

    # 2. Livello IDWT
    HL1_wm = pywt.waverec2((LL_HL2, (HL12_wm, LH12_orig, HH12_orig)), 'haar') # **CORREZIONE**
    LH1_wm = pywt.waverec2((LL_LH2, (HL22_wm, LH22_orig, HH22_orig)), 'haar') # **CORREZIONE**

    # 1. Livello IDWT (Immagine finale)
    coeffs1_wm = LL1, (HL1_wm, LH1_wm, HH1)
    watermarked_image = pywt.waverec2(coeffs1_wm, 'haar')
    
    return watermarked_image[:IMAGE_SIZE, :IMAGE_SIZE] # Ritorna solo l'array float