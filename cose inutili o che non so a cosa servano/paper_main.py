import numpy as np
import cv2
import os

# Importiamo le funzioni e le costanti dai file dedicati
from CTM_ecorp.embedding.paper_embedding_try1 import ( # Assicurati che il path sia corretto
    embedding_algorithm, arnold_cat_map, generate_pn_sequence, 
    IMAGE_SIZE, WATERMARK_SIZE, ALPHA
)
from detection.paper_detection import ( # Assicurati che il path sia corretto
    detection_function, DETECTION_THRESHOLD
)

# --- CHIAVI SEGRETE ---
ARNOLD_ITER = 3    
PN_SEED = 42       
COEFF_PER_BLOCK = 8 
TOTAL_BLOCKS = WATERMARK_SIZE**2 # 1024

# --- FUNZIONE PRINCIPALE DI GESTIONE CARTELLA ---

def process_images_in_folder(input_dir, output_dir, original_watermark, watermark_bits, pn0, pn1):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detection_results = {}
    
    # Ottieni la lista dei file
    file_list = os.listdir(input_dir)
    
    # *** NUOVO CONTROLLO ***
    if not file_list:
        print(f"ERRORE CRITICO: La cartella di input '{input_dir}' è vuota. Inserisci le immagini da elaborare.")
        return detection_results
    
    processed_count = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"wm_{filename}")
            
            print(f"\nProcessing {filename}...")
            
            # Preparazione Immagine Host
            host_img_uint8 = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            # *** CONTROLLO POTENZIATO SULLA LETTURA DELL'IMMAGINE ***
            if host_img_uint8 is None or host_img_uint8.size == 0:
                print(f"ATTENZIONE: Impossibile caricare o l'immagine {filename} è vuota/non valida. Saltando.")
                continue

            host_img_resized = cv2.resize(host_img_uint8, (IMAGE_SIZE, IMAGE_SIZE))
            host_image_float = host_img_resized.astype(np.float64)

            # 1. EMBEDDING (RESTITUISCE L'IMMAGINE FLOAT)
            watermarked_image_float = embedding_algorithm(
                host_image_float, watermark_bits, ALPHA, pn0, pn1, ARNOLD_ITER # <--- AGGIUNTO ARNOLD_ITER
            )
            
            watermarked_image_uint8 = np.clip(watermarked_image_float, 0, 255).astype(np.uint8)

            # 2. SALVATAGGIO DELL'IMMAGINE MARCATÀ
            cv2.imwrite(output_path, watermarked_image_uint8)
            print(f"Immagine con watermark salvata in: {output_path}")

            # 3. DETECTION (RESTITUISCE is_present E nc_value)
            # Ricarichiamo l'immagine per la detection
            reloaded_wm_uint8 = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            reloaded_wm_float = reloaded_wm_uint8.astype(np.float64)

            is_present, nc_value = detection_function(
                original_watermark, reloaded_wm_float, pn0, pn1, ARNOLD_ITER
            )
            
            detection_results[filename] = {
                "Watermarked_File": f"wm_{filename}",
                "NC_Value": round(nc_value, 4),
                "Watermark_Presente": is_present
            }
            status = "SÌ" if is_present else "NO"
            print(f"-> Detection Result: NC={round(nc_value, 4)} -> Rilevato: {status}")

    return detection_results

# ----------------------------------------------------------------------
# --- ESECUZIONE (SETUP E CHIAMATA) ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    INPUT_FOLDER = "images"
    OUTPUT_FOLDER = "watermarked_images"

    # 1. Preparazione delle Chiavi
    TOTAL_PN_LENGTH = TOTAL_BLOCKS * COEFF_PER_BLOCK # 8192
    
    original_watermark = np.random.randint(0, 2, size=(WATERMARK_SIZE, WATERMARK_SIZE))
    scrambled_watermark = arnold_cat_map(original_watermark, ARNOLD_ITER)
    watermark_bits = scrambled_watermark.flatten()
    
    # *** CORREZIONE: Lunghezza PN per 8192 coefficienti ***
    pn0 = generate_pn_sequence(TOTAL_PN_LENGTH, PN_SEED) 
    pn1 = generate_pn_sequence(TOTAL_PN_LENGTH, PN_SEED + 1)

    if not os.path.exists(INPUT_FOLDER):
        print(f"ERRORE: La cartella di input '{INPUT_FOLDER}' non esiste.")
        print("Crea la cartella e inserisci le immagini per procedere.")
    else:
        print(f"Inizio elaborazione immagini da '{INPUT_FOLDER}'...")
        
        results = process_images_in_folder(
            INPUT_FOLDER, OUTPUT_FOLDER, 
            original_watermark, watermark_bits, 
            pn0, pn1
        )

        # Stampa i risultati finali della detection
        print("\n" + "="*50)
        print("ESITO FINALE DELLA DETECTION PER TUTTE LE IMMAGINI")
        print("="*50)
        for filename, data in results.items():
            status = "SÌ" if data['Watermark_Presente'] else "NO"
            print(f"Immagine {filename}: Rilevato: {status} (NC={data['NC_Value']})")
        print("="*50)