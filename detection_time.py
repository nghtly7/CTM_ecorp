import os
import time
import importlib.util
import numpy as np
import cv2

BASE_DIR = "groups_prova"
NOISE_STD = 2  # deviazione standard del rumore gaussiano
TEMP_ATTACKED = "attacked_tmp.bmp"  # file temporaneo

def add_awgn(image, std=2):
    """Aggiunge rumore gaussiano bianco (AWGN)."""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def load_module(path):
    """Carica dinamicamente un file Python (.py o .pyc) come modulo."""
    spec = importlib.util.spec_from_file_location("detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    results = []

    for group_name in os.listdir(BASE_DIR):
        group_path = os.path.join(BASE_DIR, group_name)
        if not os.path.isdir(group_path):
            continue

        print(f"\n=== Gruppo: {group_name} ===")

        # trova file di detection (.py o .pyc)
        code_files = [f for f in os.listdir(group_path) if f.endswith((".py", ".pyc"))]
        if not code_files:
            print(" ⚠️ Nessun file di detection trovato, salto.")
            continue

        detector_path = os.path.join(group_path, code_files[0])
        detector = load_module(detector_path)

        # trova immagini BMP
        bmp_files = sorted([f for f in os.listdir(group_path) if f.lower().endswith(".bmp")])
        if not bmp_files:
            print(" ⚠️ Nessuna immagine BMP trovata, salto.")
            continue

        # definisci immagine originale (adatta se ne hai una specifica)
        original_image = os.path.join(group_path, bmp_files[0])

        for bmp_file in bmp_files:
            watermarked_path = os.path.join(group_path, bmp_file)
            img = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # applica rumore AWGN e salva temporaneo
            attacked = add_awgn(img, NOISE_STD)
            attacked_path = os.path.join(group_path, TEMP_ATTACKED)
            cv2.imwrite(attacked_path, attacked)

            # misura il tempo della detection (senza stampare output)
            start = time.time()
            try:
                _ = detector.detection(original_image, watermarked_path, attacked_path)
            except Exception as e:
                print(f" ⚠️ Errore in detection su {bmp_file}: {e}")
                continue
            elapsed = time.time() - start

            print(f" {bmp_file}: {elapsed:.4f}s")
            results.append((group_name, bmp_file, elapsed))

            # rimuovi il file temporaneo
            try:
                os.remove(attacked_path)
            except OSError:
                pass

    # report finale
    print("\n=== Report Finale ===")
    for group, bmp, t in results:
        print(f"{group:<20} {bmp:<20} {t:.4f}s")

if __name__ == "__main__":
    main()
