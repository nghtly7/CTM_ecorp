#!/usr/bin/env python3
"""
main_bfa.py

Usare:
  - Metti questo file nella root del progetto (stesso livello di "groups" e "original_images").
  - Struttura attesa:
      ./groups/
         gruppo_1/
            gruppo_1_uno.bmp
            gruppo_1_due.bmp
            gruppo_1_tre.bmp
            detection_gruppo_1.py   # oppure detection_gruppo_1.pyc
         gruppo_2/
            ...
      ./original_images/
         uno.bmp
         due.bmp
         tre.bmp
      ./attacks/attack_brute_force.py   # modulabile e importabile

  - Esegui: python3 main_bfa.py

Comportamento:
  - Per ogni gruppo, carica il modulo detection (qualsiasi file che inizi con "detection" e finisca in .py o .pyc).
  - Per ogni immagine nel gruppo, mappa il nome rimuovendo il prefisso "<groupname>_" e cerca quel file in original_images.
    Se non trova l'originale, userà l'immagine del gruppo come fallback.
  - Chiama attacks.attack_brute_force.find_best_attack(original_path, watermarked_path, adv_detection_func, ...)
    e salva i risultati (immagine attaccata, CSV).
"""
import os
import sys
import csv
import time
import traceback
import importlib.util
import importlib.machinery
from typing import Optional, Callable, Tuple, Dict, Any
import attacks.attack_brute_force as bf

ROOT = os.path.abspath(os.path.dirname(__file__))

# --- Configurazione (modifica se necessario) ---
GROUPS_DIR = os.path.join(ROOT, "groups_prova")
ORIGINAL_IMAGES_DIR = os.path.join(ROOT, "input_images/images")
OUT_ATT_ROOT = os.path.join(ROOT, "attacked_images")
CSV_SUMMARY = os.path.join(ROOT, "attack_summary.csv")
# path al modulo brute-force (deve essere importabile)
BF_MODULE_PATH = os.path.join(ROOT, "attacks", "attack_brute_force.py")
# --- fine config ---

# Ensure output dirs exist
os.makedirs(OUT_ATT_ROOT, exist_ok=True)

# --- Helper per caricare detection locale in modo robusto ---
def find_detection_file(group_dir: str) -> Optional[str]:
    """
    Cerca un file che comincia con 'detection' (case-insensitive) e finisce con .py o .pyc.
    Restituisce il path completo o None se non trovato.
    """
    for fname in os.listdir(group_dir):
        low = fname.lower()
        if (low.startswith("detection") and (low.endswith(".py") or low.endswith(".pyc"))):
            return os.path.join(group_dir, fname)
    return None

def load_detection_callable(path: str) -> Callable:
    """
    Carica dal file path un modulo e ritorna l'attributo 'detection' (callable),
    registrando il modulo in sys.modules per consentirne il pickle da multiprocessing.
    """
    import sys
    import importlib.util
    import importlib.machinery
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Usa un nome stabile e univoco per il modulo (pickle-safe)
    base_name = os.path.splitext(os.path.basename(path))[0]
    safe_name = f"detection_module_{base_name}"

    if path.lower().endswith(".py"):
        spec = importlib.util.spec_from_file_location(safe_name, path)
    else:
        loader = importlib.machinery.SourcelessFileLoader(safe_name, path)
        spec = importlib.util.spec_from_loader(safe_name, loader)

    mod = importlib.util.module_from_spec(spec)
    sys.modules[safe_name] = mod  # 👈 chiave: registra il modulo nel namespace globale
    spec.loader.exec_module(mod)  # type: ignore

    if not hasattr(mod, "detection"):
        raise AttributeError(f"Module {path} non espone 'detection' callable")
    det = getattr(mod, "detection")
    if not callable(det):
        raise TypeError(f"'detection' in {path} non è callable")

    return det


# --- Import del modulo brute-force (attacks.attack_brute_force) ---
# Proviamo a importarlo come modulo Python. Se BF_MODULE_PATH non è presente, proviamo a importare il package attacks.

if bf is None:
    print("Errore: il modulo attacks.attack_brute_force non è importabile. Assicurati che 'attacks/attack_brute_force.py' esista.")
    # Non usciamo subito, perché potresti voler vedere lo script; ma quando lo esegui il modulo è richiesto.
    # sys.exit(1)

# --- Utility di mapping filename ---
def map_group_to_original(group_name: str, group_img_name: str) -> str:
    """
    Esempio: group_name='gruppo_1', group_img_name='gruppo_1_uno.bmp' -> 'uno.bmp'
    Se non c'è underscore, prova a restituire basename unchanged.
    """
    base = os.path.basename(group_img_name)
    if base.startswith(group_name + "_"):
        return base[len(group_name) + 1 :]
    # altrimenti, prova a cercare il primo underscore e ritagliare fino al primo underscore
    if "_" in base:
        return base.split("_", 1)[-1]
    return base

def find_original_file(original_basename: str) -> Optional[str]:
    """
    Cerca original_basename dentro ORIGINAL_IMAGES_DIR (case-sensitive).
    """
    cand = os.path.join(ORIGINAL_IMAGES_DIR, original_basename)
    if os.path.exists(cand):
        return cand
    # try a case-insensitive search if not found (helpful su Windows)
    for f in os.listdir(ORIGINAL_IMAGES_DIR) if os.path.isdir(ORIGINAL_IMAGES_DIR) else []:
        if f.lower() == original_basename.lower():
            return os.path.join(ORIGINAL_IMAGES_DIR, f)
    return None

# --- Main pipeline ---
def main():
    start_time = time.time()

    # Validate dirs
    if not os.path.isdir(GROUPS_DIR):
        print("GROUPS_DIR non trovato:", GROUPS_DIR)
        return
    if not os.path.isdir(ORIGINAL_IMAGES_DIR):
        print("ORIGINAL_IMAGES_DIR non trovato:", ORIGINAL_IMAGES_DIR)
        # Non blockiamo l'esecuzione: userà fallback se necessario.

    # Prepare CSV
    csv_fields = ["group", "image_in_group", "original_used", "watermarked_path", "best_attack_names",
                  "best_attack_params", "best_wpsnr", "attack_success_bool", "attacked_save_path", "notes"]
    csv_f = open(CSV_SUMMARY, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    writer.writeheader()

    groups = sorted([d for d in os.listdir(GROUPS_DIR) if os.path.isdir(os.path.join(GROUPS_DIR, d))])
    if not groups:
        print("Nessun gruppo trovato in", GROUPS_DIR)
    for group in groups:
        group_dir = os.path.join(GROUPS_DIR, group)
        print("\n" + "=" * 60)
        print(f"[GROUP] Processing group: {group} ({group_dir})")

        # find detection file inside group
        det_file = find_detection_file(group_dir)
        if not det_file:
            print(f"  WARNING: nessun file di detection trovato in {group_dir}. Skipping group.")
            writer.writerow({
                "group": group,
                "image_in_group": "",
                "original_used": "",
                "watermarked_path": "",
                "best_attack_names": "",
                "best_attack_params": "",
                "best_wpsnr": 0.0,
                "attack_success_bool": False,
                "attacked_save_path": "",
                "notes": "detection_not_found"
            })
            csv_f.flush()
            continue

        try:
            detection_fn = load_detection_callable(det_file)
        except Exception as e:
            print(f"  ERRORE: impossibile caricare detection in {det_file}: {e}")
            traceback.print_exc()
            writer.writerow({
                "group": group,
                "image_in_group": "",
                "original_used": "",
                "watermarked_path": "",
                "best_attack_names": "",
                "best_attack_params": "",
                "best_wpsnr": 0.0,
                "attack_success_bool": False,
                "attacked_save_path": "",
                "notes": f"detection_load_error:{e}"
            })
            csv_f.flush()
            continue

        # find images in group (bmp/png/jpg/jpeg)
        valid_exts = (".bmp", ".png", ".jpg", ".jpeg")
        group_images = sorted([f for f in os.listdir(group_dir) if f.lower().endswith(valid_exts)])
        if not group_images:
            print(f"  Nessuna immagine trovata in {group_dir}.")
            writer.writerow({
                "group": group,
                "image_in_group": "",
                "original_used": "",
                "watermarked_path": "",
                "best_attack_names": "",
                "best_attack_params": "",
                "best_wpsnr": 0.0,
                "attack_success_bool": False,
                "attacked_save_path": "",
                "notes": "no_images"
            })
            csv_f.flush()
            continue

        # Create out dir for group
        out_group_dir = os.path.join(OUT_ATT_ROOT, group)
        os.makedirs(out_group_dir, exist_ok=True)

        # For each image in the group, map to original and run brute-force
        for gi in group_images:
            wm_path = os.path.join(group_dir, gi)
            mapped_basename = map_group_to_original(group, gi)
            orig_path = find_original_file(mapped_basename)
            if orig_path:
                print(f"  {gi} -> original: {orig_path}")
                original_used = orig_path
            else:
                print(f"  {gi} -> original not found ({mapped_basename}); using group image as fallback")
                original_used = wm_path

            # call bf.find_best_attack (must exist in attacks.attack_brute_force)
            best_attack = None
            best_wpsnr = 0.0
            attacked_save_path = ""
            attack_success_bool = False
            try:
                if bf is None:
                    raise ImportError("Modulo brute-force (attacks.attack_brute_force) non importato.")
                # find_best_attack(original_path, watermarked_path, adv_detection_func, max_workers=?)
                # A seconda dell'implementazione di bf, possono cambiare i nomi dei parametri.
                # Qui chiamiamo con i positional args più probabili.
                res = bf.find_best_attack(
                    original_path=original_used,
                    watermarked_path=wm_path,
                    adv_detection_func=detection_fn,
                    max_workers=1  # 🚀 Disabilita multiprocessing → niente pickle error
                )
                # res expected: (best_attack_dict, best_wpsnr)
                if isinstance(res, tuple) and len(res) >= 2:
                    best_attack, best_wpsnr = res[0], res[1]
                else:
                    # se find_best_attack ritorna solo best_attack, cerchiamo wpsnr nell'oggetto
                    best_attack = res
                    best_wpsnr = getattr(res, "wpsnr", 0.0) if res is not None else 0.0
            except Exception as e:
                print(f"    ERROR during find_best_attack for {gi}: {e}")
                traceback.print_exc()
                writer.writerow({
                    "group": group,
                    "image_in_group": gi,
                    "original_used": original_used,
                    "watermarked_path": wm_path,
                    "best_attack_names": "",
                    "best_attack_params": "",
                    "best_wpsnr": 0.0,
                    "attack_success_bool": False,
                    "attacked_save_path": "",
                    "notes": f"bf_error:{e}"
                })
                csv_f.flush()
                continue

            # If we have a best_attack spec, try to produce final attacked image and save
            attack_names = ""
            attack_params = ""
            if best_attack:
                attack_names = "+".join(best_attack.get("names", [])) if isinstance(best_attack, dict) else str(best_attack)
                attack_params = str(best_attack.get("params", [])) if isinstance(best_attack, dict) else ""
                try:
                    # some implementations expose an 'attacks' function to apply names+params to produce image
                    if hasattr(bf, "attacks"):
                        final_img = bf.attacks(wm_path, best_attack.get("names", []), best_attack.get("params", []))
                        # bf.attacks may return numpy array; save using OpenCV if available, else try pillow
                        try:
                            import cv2
                            import numpy as np
                            out_name = f"{os.path.splitext(gi)[0]}_attacked.bmp"
                            attacked_save_path = os.path.join(out_group_dir, out_name)
                            # if final_img is array, write
                            cv2.imwrite(attacked_save_path, final_img)
                        except Exception:
                            # fallback: try PIL
                            try:
                                from PIL import Image
                                out_name = f"{os.path.splitext(gi)[0]}_attacked.bmp"
                                attacked_save_path = os.path.join(out_group_dir, out_name)
                                if hasattr(final_img, "save"):
                                    final_img.save(attacked_save_path)
                                else:
                                    # assume numpy array but no cv2
                                    import numpy as np
                                    im = Image.fromarray(np.uint8(final_img))
                                    im.save(attacked_save_path)
                            except Exception as e2:
                                print("      WARNING: non sono riuscito a salvare l'immagine attaccata:", e2)
                                attacked_save_path = ""
                    else:
                        # if no bf.attacks available, maybe best_attack already contains path or image
                        # try common fallbacks:
                        if isinstance(best_attack, dict) and "attacked_image" in best_attack:
                            attacked_save_path = os.path.join(out_group_dir, f"{os.path.splitext(gi)[0]}_attacked.bmp")
                            try:
                                import cv2
                                cv2.imwrite(attacked_save_path, best_attack["attacked_image"])
                            except Exception:
                                try:
                                    from PIL import Image
                                    im = Image.fromarray(best_attack["attacked_image"])
                                    im.save(attacked_save_path)
                                except Exception:
                                    attacked_save_path = ""
                except Exception as e:
                    print(f"    WARNING: errore nel generare/salvare immagine attaccata per {gi}: {e}")
                    traceback.print_exc()

                # Decide success boolean using best_wpsnr (heuristica) or content of best_attack
                try:
                    # if best_attack contains 'found' flag (0==success in alcuni codici), use it
                    if isinstance(best_attack, dict) and "found" in best_attack:
                        attack_success_bool = (best_attack.get("found", 1) == 0)
                    else:
                        # fallback: success if wpsnr >= 35 (configurabile)
                        attack_success_bool = float(best_wpsnr) >= 35.0
                except Exception:
                    attack_success_bool = False

            # Write CSV row
            writer.writerow({
                "group": group,
                "image_in_group": gi,
                "original_used": original_used,
                "watermarked_path": wm_path,
                "best_attack_names": attack_names,
                "best_attack_params": attack_params,
                "best_wpsnr": round(float(best_wpsnr), 4) if best_wpsnr else 0.0,
                "attack_success_bool": bool(attack_success_bool),
                "attacked_save_path": attacked_save_path,
                "notes": ""
            })
            csv_f.flush()

        print(f"[{group}] Finished. Attacked images (if any) in: {out_group_dir}")

    csv_f.close()
    elapsed = time.time() - start_time
    print("\nAll groups processed. Summary saved to:", CSV_SUMMARY)
    print("Elapsed (s):", round(elapsed, 1))


if __name__ == "__main__":
    main()
