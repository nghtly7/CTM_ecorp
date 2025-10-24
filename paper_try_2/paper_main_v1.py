#!/usr/bin/env python3
import os
import sys
import cv2
import subprocess   # per chiamare il tuning

# assicuriamoci di poter importare wpsnr dalla cartella superiore
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(ROOT_DIR)


from wpsnr import wpsnr
# from paper_try_2.paper_embedding_v2 import embedding_v2
# from paper_try_2.paper_detection_v2 import detection_v2     # la tua detection file (single file)
from paper_try_2.paper_embedding_v1_stronger import embedding
from paper_try_2.paper_detection_v1_stronger import detection
from awgn_attack import attacks           # lo script AWGN che hai aggiunto

# cartelle (relativamente alla posizione di questo main)
INPUT_DIR = os.path.join(ROOT_DIR, "images")
WM_DIR    = os.path.join(ROOT_DIR, "..", "watermarked_images")
ATT_DIR   = os.path.join(ROOT_DIR, "..", "attacked_images")   # richiesta: salvare attaccate qui


# tuning script path (modifica se lo tieni altrove)
TUNING_SCRIPT = os.path.join(ROOT_DIR, "paper_try_2/tuning_per_image_v2.py")

# parametri AWGN per il test (modificabili)
AWGN_PARAMS = {
    "sigma_start": 0.5,
    "sigma_end": 100.0,
    "n_steps": 20,
    "seed": 12345,
    # "out_dir" verrà impostato dinamicamente per ogni immagine su ATT_DIR
}

# soglia per considerare un attacco "successo" (regole): WPSNR >= 35 e watermark distrutto (present==0)
WPSNR_SUCCESS_THRESH = 25.0

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
        print("⚠️ Nessuna immagine trovata nella cartella 'images/'")
        return

    print(f"📌 Trovate {len(images)} immagini. Avvio pipeline (embed -> attack -> detect)...\n")
    
    wpsnr_threshold = 54.0
    count_wpsnr_ge_58 = 0

    for img_name in images:
        input_path  = os.path.join(INPUT_DIR, img_name)
        wm_path     = os.path.join(WM_DIR, img_name)

        print(f"🔹 Processing: {img_name}")

        # 1) EMBEDDING (uses embedding(input_path) and saves watermark internally if generated)
        try:
            # se la tua embedding supporta args, puoi passarle qui; per ora chiamiamo semplicemente embedding(input_path)
            Iw = embedding(input_path)   # ritorna array uint8
        except Exception as e:
            print(f"  ERROR embedding {img_name}: {e}")
            continue

        original_basename = os.path.basename(input_path)
        try:
            wpsnr_val = wpsnr(cv2.imread(input_path, cv2.IMREAD_GRAYSCALE), Iw)
        except Exception as e:
            print(f"  ERROR computing WPSNR for {img_name}: {e}")
            wpsnr_val = float('nan')
        print(f"    Watermark embedded. WPSNR between original and watermarked: {wpsnr_val:.2f} dB")

        try:
            if not (wpsnr_val is None) and (not (wpsnr_val != wpsnr_val)):  # check NaN
                if wpsnr_val >= wpsnr_threshold:
                    count_wpsnr_ge_58 += 1
        except Exception:
            pass
        
        # save watermarked image
        cv2.imwrite(wm_path, Iw)
        
        # 2) Attack: AWGN progressive -> save into per-image subfolder inside ATT_DIR
        #    create a dedicated folder per original filename to avoid name collisions
        base_noext = os.path.splitext(img_name)[0]
        per_img_att_dir = os.path.join(ATT_DIR, base_noext)
        ensure_dir(per_img_att_dir)

        awgn_params = AWGN_PARAMS.copy()
        awgn_params["out_dir"] = per_img_att_dir

        try:
            attacked_paths = attacks(wm_path, "awgn", awgn_params)
        except Exception as e:
            print(f"  ERROR during AWGN attacks for {img_name}: {e}")
            continue
            
        # 3) For each attacked image run detection(original, watermarked, attacked)
        #    and decide if the attack "succeeded" (present==0 and wpsnr >= 35)
        success_found = False
        for attacked_file in attacked_paths:
            try:
                present_flag, wpsnr_val = detection(input_path, wm_path, attacked_file)
            except Exception as e:
                print(f"    ERROR detection on {os.path.basename(attacked_file)}: {e}")
                continue

            # parse sigma from filename for nicer printing if present
            fname = os.path.basename(attacked_file)
            # attempt to extract "sigmaXX" substring
            sigma_str = ""
            if "sigma" in fname:
                try:
                    # find substring like sigmaXX.XX
                    idx = fname.index("sigma")
                    sigma_str = fname[idx: idx + fname[idx:].find(".")]
                except Exception:
                    sigma_str = ""

            status = "PRESENT" if present_flag == 1 else "DESTROYED"
            success = (present_flag == 0 and wpsnr_val >= WPSNR_SUCCESS_THRESH)

            print(f"    -> {fname} : detection={status}, WPSNR={wpsnr_val:.2f} dB, success={success}")

            if success:
                success_found = True
                # non obbligatorio: se vuoi fermarti al primo successo, decommenta la prossima riga
                # break

        if success_found:
            print(f"  ==> ATTACK SUCCESSFUL for at least one noise level on {img_name}\n")
        else:
            print(f"  ==> No successful attack found (wpsnr >= {WPSNR_SUCCESS_THRESH} & watermark destroyed) for {img_name}\n")
        
    print("🎯 Pipeline completata.")
    print(f"📊 Immagini con WPSNR >= {wpsnr_threshold:.1f} dB: {count_wpsnr_ge_58}/{len(images)}")
    

if __name__ == "__main__":
    main(sys.argv)
