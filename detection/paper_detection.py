import numpy as np
import pywt
from scipy.fft import dctn

# --- COSTANTI (Devono essere coerenti con watermark_embedder.py) ---
IMAGE_SIZE = 512
WATERMARK_SIZE = 32 # 32x32 = 1024 bit (FISSO)
BLOCK_SIZE = 4
DETECTION_THRESHOLD = 0.80

# --- FUNZIONI DI UTILITY ---
def arnold_cat_map(img, iterations):
    """Implementazione semplificata della Arnold Cat Map per scramble/unscramble."""
    if iterations > 0:
        return np.roll(img, iterations, axis=(0, 1))
    else:
        return np.roll(img, -iterations, axis=(0, 1)) 

# --- FUNZIONE DI ESTREZIONE REVISIONATA ---

def extraction_algorithm(watermarked_image, pn0_full, pn1_full, arnold_iter):
    """Procedura di estrazione DWT-DCT blind su QUATTRO bande."""
    
    # DWT Livello 1
    coeffs1 = pywt.wavedec2(watermarked_image, 'haar', level=1)
    LL1, (HL1, LH1, HH1) = coeffs1
    
    # DWT Livello 2 su HL1 e LH1
    coeffs_hl2 = pywt.wavedec2(HL1, 'haar', level=1)
    LL_HL2, (HL12, LH12, HH12) = coeffs_hl2
    coeffs_lh2 = pywt.wavedec2(LH1, 'haar', level=1)
    LL_LH2, (HL22, LH22, HH22) = coeffs_lh2 

    # DWT Livello 3
    LL_HL12_3, (HL13, LH13, HH13) = pywt.wavedec2(HL12, 'haar', level=1)
    LL_LH12_3, (HL23_T, LH13_T, HH23_T) = pywt.wavedec2(LH12, 'haar', level=1) 
    LL_HL22_3, (HL23, LH23_T_2, HH23_T_2) = pywt.wavedec2(HL22, 'haar', level=1) 
    LL_LH22_3, (HL33_T, LH23, HH33_T) = pywt.wavedec2(LH22, 'haar', level=1)

    # I target sono HL13, LH13_T, HL23, LH23
    bands_to_extract = [HL13, LH13_T, HL23, LH23]
    
    extracted_watermark_bits = []
    num_pn_coeffs = 8 
    watermark_idx = 0

    for target_band in bands_to_extract:
        h, w = target_band.shape
        num_blocks_h = h // BLOCK_SIZE
        num_blocks_w = w // BLOCK_SIZE

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                
                block = target_band[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
                dct_block = dctn(block, type=2, norm='ortho')
                
                # Estrazione dei coefficienti di banda media
                mid_band_coeffs = []
                mid_band_indices = list(zip(*np.unravel_index(np.arange(2, 2 + num_pn_coeffs), (BLOCK_SIZE, BLOCK_SIZE))))
                
                for r, c in mid_band_indices:
                    mid_band_coeffs.append(dct_block[r, c])
                mid_band_coeffs = np.array(mid_band_coeffs)
                
                # Truncamento della sequenza PN per il blocco corrente
                pn0_block = pn0_full[watermark_idx * num_pn_coeffs : (watermark_idx + 1) * num_pn_coeffs]
                pn1_block = pn1_full[watermark_idx * num_pn_coeffs : (watermark_idx + 1) * num_pn_coeffs]
                
                # Correlazione (stima del bit)
                corr_pn0 = np.dot(mid_band_coeffs, pn0_block)
                corr_pn1 = np.dot(mid_band_coeffs, pn1_block)
                
                extracted_bit = 0 if corr_pn0 > corr_pn1 else 1
                extracted_watermark_bits.append(extracted_bit)
                
                watermark_idx += 1
    
    # ASSICURAZIONE: estraiamo esattamente 1024 bit
    if len(extracted_watermark_bits) != WATERMARK_SIZE**2:
        raise ValueError(f"Estrazione fallita: trovati {len(extracted_watermark_bits)} bit, attesi 1024.")

    # Ricostruzione finale (Arnold inversa)
    extracted_scrambled_wm = np.array(extracted_watermark_bits).reshape(WATERMARK_SIZE, WATERMARK_SIZE)
    extracted_wm = arnold_cat_map(extracted_scrambled_wm, -arnold_iter)
    
    return extracted_wm

# --- FUNZIONE DI DETECTION ---

def normalized_correlation(W_original, W_extracted):
    """Calcola la Normalized Correlation (NC) tra due matrici binarie."""
    W_flat = W_original.flatten()
    W_hat_flat = W_extracted.flatten()
    
    numerator = np.sum(W_flat * W_hat_flat)
    norm_W = np.sqrt(np.sum(W_flat**2))
    norm_W_hat = np.sqrt(np.sum(W_hat_flat**2))
    
    if norm_W == 0 or norm_W_hat == 0:
        return 0.0
    
    return numerator / (norm_W * norm_W_hat)

def detection_function(original_watermark, watermarked_attacked_image, pn0_full, pn1_full, arnold_iter, detection_threshold=DETECTION_THRESHOLD):
    """Verifica la presenza del watermark nell'immagine attaccata."""
    
    extracted_watermark = extraction_algorithm(watermarked_attacked_image, pn0_full, pn1_full, arnold_iter)
    nc_value = normalized_correlation(original_watermark, extracted_watermark)
    
    is_present = nc_value >= detection_threshold
    
    # Ritorna l'esito SI/NO (booleano) e il valore NC per l'analisi
    return is_present, nc_value