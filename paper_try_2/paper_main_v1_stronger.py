#!/usr/bin/env python3
import os
import sys
import cv2
import subprocess   # per chiamare il tuning
import multiprocessing

# assicuriamoci di poter importare wpsnr dalla cartella superiore
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(ROOT_DIR)


from wpsnr import wpsnr
from paper_try_2.paper_embedding_v1_stronger import embedding
from paper_try_2.paper_detection_v1_stronger import detection
# import del brute-force attacks engine
import attack_brute_force as bf

# cartelle (relativamente alla posizione di questo main)
INPUT_DIR = os.path.join(ROOT_DIR, "images")
WM_DIR    = os.path.join(ROOT_DIR, "..", "watermarked_images")
ATT_DIR   = os.path.join(ROOT_DIR, "..", "attacked_images")   # richiesta: salvare attaccate qui

# tuning script path (modifica se lo tieni altrove)
TUNING_SCRIPT = os.path.join(ROOT_DIR, "paper_try_2/tuning_per_image_v2.py")

# soglia per considerare un attacco "successo" (regole): WPSNR >= 35 e watermark distrutto (present==0)
WPSNR_SUCCESS_THRESH = 35.0

# ADV group name usato per il log CSV (modifica se necessario)
ADV_GROUP_NAME = "adversary"   # cambia con il nome del gruppo adversario se lo conosci
OUR_GROUP_NAME = "ecorp"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    
def run_tuning_per_image(script_path):
    if not os.path.isfile(script_path):
        print(f"[TUNING] tuning_per_image non trovato: {script_path}")
        return 2
    print(f"[TUNING] Avvio tuning per-immagine: {script_path}")
    import subprocess
    proc = subprocess.run([sys.executable, script_path], cwd=ROOT_DIR)
    print(f"[TUNING] tuning per-immagine terminato: returncode={proc.returncode}")
    return proc.returncode
    
# tuing part
def run_tuning(tuning_script_path):
    """Esegue il tuning.py come processo esterno usando lo stesso interprete Python."""
    if not os.path.isfile(tuning_script_path):
        print(f"[TUNING] tuning script non trovato: {tuning_script_path}")
        return 2
    print(f"[TUNING] Avvio tuning: {tuning_script_path}")
    # esegui con lo stesso interprete usato per questo processo
    proc = subprocess.run([sys.executable, tuning_script_path], cwd=ROOT_DIR)
    print(f"[TUNING] tuning finished with returncode = {proc.returncode}")
    return proc.returncode

def main(argv):
    # supporta arg '--tune' per lanciare tuning.py e uscire
    if len(argv) > 1 and argv[1] in ("--tune", "tune"):
        rc = run_tuning(TUNING_SCRIPT)
        if rc != 0:
            print("[TUNING] Errore o exit code diverso da 0. Controlla tuning_output/ e log.")
        return
    
    # supporta arg '--tune-per-image' per tuning PER IMMAGINE e uscita immediata
    if len(argv) > 1 and argv[1] in ("--tune-per-image", "tune-per-image"):
        rc = run_tuning_per_image(TUNING_SCRIPT)
        if rc != 0:
            print("[TUNING] Errore o exit code diverso da 0. Controlla tuning_output/per_image/ e log.")
        return
    
    ensure_dir(WM_DIR)
    ensure_dir(ATT_DIR)

    valid_exts = (".bmp")
    if not os.path.isdir(INPUT_DIR):
        print(f"Input dir not found: {INPUT_DIR}")
        return

    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_exts)]
    if len(images) == 0:
        print("⚠️ Nessuna immagine trovata nella cartella 'sample-images/'")
        return

    print(f"📌 Trovate {len(images)} immagini. Avvio pipeline (embed -> brute-force attack -> detect)...\n")
    
    wpsnr_threshold = 54.0
    count_wpsnr_ge_thr = 0

    # numero di worker per il brute-force (lasciare default CPU count)
    NUM_WORKERS = multiprocessing.cpu_count() or 4

    for img_name in images:
        input_path  = os.path.join(INPUT_DIR, img_name)
        wm_path     = os.path.join(WM_DIR, img_name)

        print(f"🔹 Processing: {img_name}")

        # 1) EMBEDDING (uses embedding(input_path) and saves watermark internally if generated)
        try:
            Iw = embedding(input_path)   # ritorna array uint8
        except Exception as e:
            print(f"  ERROR embedding {img_name}: {e}")
            continue

        try:
            wpsnr_val = wpsnr(cv2.imread(input_path, cv2.IMREAD_GRAYSCALE), Iw)
        except Exception as e:
            print(f"  ERROR computing WPSNR for {img_name}: {e}")
            wpsnr_val = float('nan')
        print(f"    Watermark embedded. WPSNR between original and watermarked: {wpsnr_val:.2f} dB")

        try:
            if not (wpsnr_val is None) and (not (wpsnr_val != wpsnr_val)):  # check NaN
                if wpsnr_val >= wpsnr_threshold:
                    count_wpsnr_ge_thr += 1
        except Exception:
            pass
        
        # save watermarked image
        cv2.imwrite(wm_path, Iw)

        # ---------------------------------------------------------------
        # 2) BRUTE-FORCE ATTACK: usa attack_brute_force.find_best_attack()
        #    - la funzione richiede: original_path, watermarked_path, adv_detection_func
        #    - adv_detection_func deve avere signature (original, watermarked, attacked) -> (found_flag, wpsnr)
        # ---------------------------------------------------------------
        print("  Avvio brute-force attack search (potrebbe impiegare tempo)...")
        try:
            best_attack, best_wpsnr = bf.find_best_attack(
                original_path=input_path,
                watermarked_path=wm_path,
                adv_detection_func=detection,
                max_workers=NUM_WORKERS
            )
        except Exception as e:
            print(f"  ERRORE durante find_best_attack per {img_name}: {e}")
            best_attack, best_wpsnr = None, 0.0

        if best_attack:
            print(f"  -> Miglior attacco trovato: {best_attack['names']} {best_attack['params']} (WPSNR={best_wpsnr:.2f})")
            # genera immagine attaccata finale e salvala in ATT_DIR
            try:
                attacked_img = bf.attacks(wm_path, best_attack["names"], best_attack["params"])
                out_fname = f"{os.path.splitext(img_name)[0]}_attacked.bmp"
                out_path = os.path.join(ATT_DIR, out_fname)
                cv2.imwrite(out_path, attacked_img)
                print(f"  Attacked image salvata in: {out_path}")
            except Exception as e:
                print(f"  ERRORE generazione immagine attaccata: {e}")

            # registra risultato su CSV
            try:
                csv_result = os.path.join(ATT_DIR, "bruteforce_results.csv")
                bf.save_attack_to_log(ADV_GROUP_NAME, img_name, best_attack, best_wpsnr, csv_result)
            except Exception as e:
                print(f"  ERRORE salvataggio log CSV: {e}")
        else:
            print(f"  Nessun attacco 'successo' trovato per {img_name} (criterio success = destroyed & WPSNR >= {WPSNR_SUCCESS_THRESH}).")

        print("")  # blank line tra immagini

    print("🎯 Pipeline completata.")
    print(f"📊 Immagini con WPSNR >= {wpsnr_threshold:.1f} dB: {count_wpsnr_ge_thr}/{len(images)}")
    

if __name__ == "__main__":
    main(sys.argv)
