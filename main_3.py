#!/usr/bin/env python3
import os
import cv2
import shutil

from wpsnr import wpsnr

from final_strategy.ecorp_embedding_v2 import embedding
from final_strategy.ecorp_detection_v2 import detection

from attacks.awgn_attack import attacks as awgn_attacks
from attacks.blur_attack import attacks as blur_attacks
from attacks.median_attack import attacks as median_attacks
from attacks.jpeg_attack import attacks as jpeg_attacks

# cartelle (relativamente alla posizione di questo main)
INPUT_DIR = os.path.join("input_images/images")
# cartella dove mettere i watermarked generati da embedding (o dove copiamo quelli forniti)
WM_DIR    = os.path.join("watermarked_images")
# cartella opzionale: immagini già watermarkate fornite dall'utente — se presente per un file,
# salta la fase di embedding e usa questa immagine come wm_path
WM_INPUT_DIR = os.path.join("watermarked_input")

ATT_DIR   = os.path.join("attacked_images")   # qui mettiamo sottocartelle per AWGN e BLUR


# parametri AWGN per il test (modificabili)
AWGN_PARAMS_TEMPLATE = {
    "sigma_start": 80.0,
    "sigma_end": 150.0,
    "n_steps": 15,
    "seed": 12345,
    # "out_dir" verrà impostato dinamicamente per ogni immagine
}

# parametri BLUR (applied independently on the watermarked image)
BLUR_PARAMS_TEMPLATE = {
    "sigma_start": 1.0,
    "sigma_end": 20.0,
    "n_steps": 12,
    # "out_dir" verrà impostato dinamicamente per ogni immagine
}

MEDIAN_PARAMS = {
    "k_start": 3.0,
    "k_end": 40.0,
    "n_steps": 12,
    #"out_dir": per_img_att_dir_median  # impostala dinamicamente per immagine
}

JPEG_PARAMS = {
    "q_start": 30.0,
    "q_end": 1.0,
    "n_steps": 6,
    #"out_dir": per_img_att_dir_jpeg,  # imposta dinamicamente
    "force_gray": True
}

# soglia per considerare un attacco "successo" (regole): WPSNR >= 35 e watermark distrutto (present==0)
WPSNR_SUCCESS_THRESH = 35.0

# soglia richiesta dall'utente per il counter (contare quante volte il watermark viene TROVATO
# in attacchi con wpsnr < 25)
WPSNR_COUNTER_THRESH = 25.0

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():

    ensure_dir(WM_DIR)
    ensure_dir(ATT_DIR)
    # non è obbligatorio che l'utente abbia una cartella di watermarked input:
    # se non esiste, la funzione continuerà a generare i watermark via embedding
    # ma la creiamo comunque così l'utente sa dove metterli
    ensure_dir(WM_INPUT_DIR)

    valid_exts = (".bmp")
    if not os.path.isdir(INPUT_DIR):
        print(f"Input dir not found: {INPUT_DIR}")
        return

    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_exts)]
    if len(images) == 0:
        print("⚠️ Nessuna immagine trovata nella cartella 'input_images/images'")
        return

    print(f"📌 Trovate {len(images)} immagini. Avvio pipeline (embed -> AWGN (separato) -> BLUR (separato) -> detect)...\n")

    # Contatore totale su tutte le immagini per il criterio richiesto
    total_found_with_low_wpsnr = 0
    total_attacks_with_low_wpsnr = 0  # opzionale: conta quanti attacchi totali avevano wpsnr < threshold

    for img_name in images:
        input_path  = os.path.join(INPUT_DIR, img_name)
        wm_path     = os.path.join(WM_DIR, img_name)

        print(f"🔹 Processing: {img_name}")

        # ---------------------------
        # Decide se SKIPPARE embedding (se esiste una watermarked image fornita dall'utente)
        # ---------------------------
        provided_wm_candidate = os.path.join(WM_INPUT_DIR, img_name)
        skip_embedding = False
        if os.path.isfile(provided_wm_candidate):
            # user provided a watermarked image with same filename in WM_INPUT_DIR:
            # use that image as wm_path and copy it into WM_DIR for record
            try:
                shutil.copyfile(provided_wm_candidate, wm_path)
                skip_embedding = True
                print("    -> Trovata immagine watermarkata in 'watermarked_input/' - salto embedding e uso quella immagine.")
            except Exception as e:
                print(f"    ERROR copying provided watermarked image: {e}")
                print("    -> Procedo con embedding (fallback).")
                skip_embedding = False

        # ---------------------------
        # 1) EMBEDDING (solo se non abbiamo un'immagine watermarkata fornita)
        # ---------------------------
        if not skip_embedding:
            try:
                Iw = embedding(input_path)   # ritorna array uint8 512x512
            except Exception as e:
                print(f"  ERROR embedding {img_name}: {e}")
                continue

            # compute and print WPSNR between original e watermarked
            try:
                orig_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                wpsnr_val = wpsnr(orig_gray, Iw)
            except Exception as e:
                print(f"  ERROR computing WPSNR per {img_name}: {e}")
                wpsnr_val = float('nan')
            print(f"    Watermark embedded. WPSNR between original and watermarked: {wpsnr_val:.2f} dB")

            # save watermarked image (in WM_DIR)
            try:
                cv2.imwrite(wm_path, Iw)
            except Exception as e:
                print(f"    ERROR saving watermarked image {wm_path}: {e}")
                continue
        else:
            # Se abbiamo saltato embedding: carichiamo l'immagine fornita e calcoliamo WPSNR rispetto all'originale
            try:
                orig_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                provided_wm_gray = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
                wpsnr_val = wpsnr(orig_gray, provided_wm_gray)
            except Exception as e:
                print(f"  ERROR computing WPSNR per {img_name} (provided wm): {e}")
                wpsnr_val = float('nan')
            print(f"    Usata immagine watermarkata fornita. WPSNR between original and provided watermarked: {wpsnr_val:.2f} dB")

        # ---------------------------
        # (il resto del flusso con attacchi e detection era commentato nel file originale;
        # non ho modificato la logica degli attacchi: se vuoi che esegua gli attacchi
        # abilita / decommenta la sezione sotto)
        # ---------------------------

        """"""
        base_noext = os.path.splitext(img_name)[0]
        per_img_att_dir_awgn = os.path.join(ATT_DIR, base_noext, "awgn")
        ensure_dir(per_img_att_dir_awgn)

        awgn_params = AWGN_PARAMS_TEMPLATE.copy()
        awgn_params["out_dir"] = per_img_att_dir_awgn

        try:
            awgn_attacked_paths = awgn_attacks(wm_path, "awgn", awgn_params)
        except Exception as e:
            print(f"  ERROR during AWGN attacks for {img_name}: {e}")
            awgn_attacked_paths = []

        # ---------------------------
        # 3) BLUR attacks on the WATERMARKED image (independent)
        # ---------------------------
        per_img_att_dir_blur = os.path.join(ATT_DIR, base_noext, "blur")
        ensure_dir(per_img_att_dir_blur)

        blur_params = BLUR_PARAMS_TEMPLATE.copy()
        blur_params["out_dir"] = per_img_att_dir_blur

        try:
            blur_attacked_paths = blur_attacks(wm_path, "blur", blur_params)
        except Exception as e:
            print(f"  ERROR during BLUR attacks for {img_name}: {e}")
            blur_attacked_paths = []
            
        # ---------------------------
        # 4) MEDIAN FILTER attacks on the WATERMARKED image (independent)
        # ---------------------------
        per_img_att_dir_median = os.path.join(ATT_DIR, base_noext, "median")
        ensure_dir(per_img_att_dir_median)
        median_params = MEDIAN_PARAMS.copy()
        median_params["out_dir"] = per_img_att_dir_median
        try:
            median_attacked_paths = median_attacks(wm_path, "median", median_params)
        except Exception as e:
            print(f"  ERROR durante MEDIAN attacks per {img_name}: {e}")
            median_attacked_paths = []
            
        # ---------------------------
        # 5) JPEG COMPRESSION attacks on the WATERMARKED image (independent)
        # ---------------------------
        per_img_att_dir_jpeg = os.path.join(ATT_DIR, base_noext, "jpeg")
        ensure_dir(per_img_att_dir_jpeg)
        jpeg_params = JPEG_PARAMS.copy()
        jpeg_params["out_dir"] = per_img_att_dir_jpeg
        try:
            jpeg_attacked_paths = jpeg_attacks(wm_path, "jpeg", jpeg_params)
        except Exception as e:
            print(f"  ERROR during JPEG attacks for {img_name}: {e}")
            jpeg_attacked_paths = []
        
        # ---------------------------
        # LAST) RUN DETECTION ON ALL ATTACKED IMAGES (AWGN-only and BLUR-only)
        # ---------------------------
        all_attacked = []
        all_attacked.extend(awgn_attacked_paths)
        # all_attacked.extend(blur_attacked_paths)
        # all_attacked.extend(median_attacked_paths)
        # all_attacked.extend(jpeg_attacked_paths)


        if len(all_attacked) == 0:
            print("  -> No attacked images generated for this input, skipping detection.\n")
            continue

        success_found = False

        # contatore per immagine: quante volte il watermark è stato TROVATO quando wpsnr < WPSNR_COUNTER_THRESH
        per_image_found_with_low_wpsnr = 0
        per_image_attacks_with_low_wpsnr = 0  # opzionale: quanti attacchi su questa immagine avevano wpsnr < threshold

        for attacked_file in all_attacked:
            try:
                present_flag, wpsnr_att = detection(input_path, wm_path, attacked_file)
            except Exception as e:
                print(f"    ERROR detection on {os.path.basename(attacked_file)}: {e}")
                continue

            fname = os.path.basename(attacked_file)
            status = "PRESENT" if present_flag == 1 else "DESTROYED"
            success = (present_flag == 0 and wpsnr_att >= WPSNR_SUCCESS_THRESH)
            print(f"    -> {fname} : detection={status}, WPSNR={wpsnr_att:.2f} dB, success={success}")

            # Se l'attacco ha WPSNR < soglia per il counter, incrementiamo i contatori opportuni
            if wpsnr_att < WPSNR_COUNTER_THRESH:
                per_image_attacks_with_low_wpsnr += 1
                total_attacks_with_low_wpsnr += 1
                if present_flag == 1:
                    per_image_found_with_low_wpsnr += 1
                    total_found_with_low_wpsnr += 1

            if success:
                success_found = True
                # se vuoi fermarti al primo successo, puoi decommentare la prossima riga
                # break
        
        

        # report per immagine (incluso contatore richiesto)
        print(f"    >> Per {img_name}: attacchi con WPSNR < {WPSNR_COUNTER_THRESH:.1f} dB: {per_image_attacks_with_low_wpsnr}, "
              f"di cui watermark TROVATO: {per_image_found_with_low_wpsnr}")

        if success_found:
            print(f"  ==> ATTACK SUCCESSFUL (AWGN o BLUR) for at least one level on {img_name}\n")
        else:
            print(f"  ==> No successful attack found (wpsnr >= {WPSNR_SUCCESS_THRESH} & watermark destroyed) per {img_name}\n")
        """"""
    # Report totale al termine della pipeline
    print("🎯 Pipeline completata.")
    print(f"🔢 Totale attacchi con WPSNR < {WPSNR_COUNTER_THRESH:.1f} dB: {total_attacks_with_low_wpsnr}")
    print(f"🔢 Totale casi in cui il watermark È STATO TROVATO con WPSNR < {WPSNR_COUNTER_THRESH:.1f} dB: {total_found_with_low_wpsnr}")


if __name__ == "__main__":
    main()
