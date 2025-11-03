#!/usr/bin/env python3
"""
main.py - test multiple embedding/detection strategies and attack them with attack_brute_force.py

Output:
 - watermarked images saved under watermarked_images/<strategy>/
 - attacked images saved under attacked_images/<strategy>/<imagename>/
 - CSV summary: attack_summary.csv (root)
"""

import os
import sys
import cv2
import time
import csv
import traceback
from typing import Callable, Optional, Tuple, Dict, Any

# try import wpsnr helper (optional, used for logging)
try:
    from wpsnr import wpsnr as wpsnr_fn
except Exception:
    wpsnr_fn = None

# ensure project root is importable
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# import brute force engine
try:
    import attacks.attack_brute_force as bf
except Exception as e:
    raise RuntimeError(f"Impossibile importare attack_brute_force.py: {e}")



# ----------------------
# Config (modifica se vuoi)
# ----------------------
INPUT_DIR = os.path.join(ROOT, "input_images/sample-images")  # dove prendere le immagini originali (.bmp)
OUT_WM_ROOT = os.path.join(ROOT, "watermarked_images")  # salvataggio watermarked per strategia
OUT_ATT_ROOT = os.path.join(ROOT, "attacked_images")    # salvataggio attacked per strategia+image (debug)
CSV_SUMMARY = os.path.join(ROOT, "attack_summary.csv")

# success criteria used by attack_brute_force (kept aligned): WPSNR >= 35 and found==0
# (the brute-force engine also uses this threshold internally)
WPSNR_SUCCESS_THRESH = 35.0

# strategies to try: list of tuples (name, embedding_import_path, detection_import_path)
# embedding_import_path should point to a module exporting `embedding`
# detection_import_path should point to a module exporting `detection`
# if detection_import_path is None we try to import detection from same module as embedding
STRATEGIES_CANDIDATES = [
    # ("final_strategy", "final_strategy.ecorp_embedding_v2", "final_strategy.ecorp_detection_v2"),
    ("final_strategy", "final_strategy.ecorp_embedding_v2", "final_strategy.ecorp_detection_v2"),


]

# ensure output folders
os.makedirs(OUT_WM_ROOT, exist_ok=True)
os.makedirs(OUT_ATT_ROOT, exist_ok=True)

# ----------------------
# Helpers: dynamic import and safe wrappers
# ----------------------
import importlib
import inspect

def try_import_callable(module_path: str, func_name: str):
    """
    Try to import a callable (func_name) from module_path.
    Returns the callable or None.
    """
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        print(f"[IMPORT] cannot import module '{module_path}': {e}")
        return None
    if hasattr(mod, func_name):
        fn = getattr(mod, func_name)
        if callable(fn):
            return fn
    print(f"[IMPORT] module '{module_path}' has no callable '{func_name}'")
    return None

def make_embedding_wrapper(embed_fn) -> Callable[[str,str], Any]:
    """
    Return wrapper that calls embed_fn robustly.
    embed_fn may accept (input_path) or (input_path, input2).
    We always call embed_fn(input_path, watermark_file) if it accepts 2 args;
    otherwise call embed_fn(input_path).
    """
    sig = None
    try:
        sig = inspect.signature(embed_fn)
    except Exception:
        pass

    def wrapper(input_path: str, watermark_file: str) -> Any:
        # prefer two-arg call if supported
        try:
            if sig is not None and len(sig.parameters) >= 2:
                return embed_fn(input_path, watermark_file)
            else:
                return embed_fn(input_path)
        except TypeError:
            # fallback try both ways
            try:
                return embed_fn(input_path, watermark_file)
            except Exception:
                return embed_fn(input_path)
    return wrapper

# ----------------------
# Instead of a closure, use a top-level picklable proxy callable
# ----------------------
import importlib

class DetectorProxy:
    """
    Picklable proxy that stores module path (string) and lazily loads the
    actual detection function inside worker processes. Instances are picklable
    because they only carry simple attributes (module_path).
    """
    def __init__(self, detection_module_path: str):
        self.detection_module_path = detection_module_path
        self._fn = None

    def _ensure_loaded(self):
        # Load on first use (works both in main process and in worker)
        if self._fn is None:
            mod = importlib.import_module(self.detection_module_path)
            if not hasattr(mod, "detection"):
                raise ImportError(f"module {self.detection_module_path} has no 'detection' callable")
            self._fn = getattr(mod, "detection")

    def __call__(self, original_path: str, watermarked_path: str, attacked_path: str):
        """
        Delegate to actual detection(original_path, watermarked_path, attacked_path)
        """
        self._ensure_loaded()
        return self._fn(original_path, watermarked_path, attacked_path)

# keep the previous try_import_callable (we can reuse it to check modules), but
# change discover_strategies to pass module path strings and wrap detection with DetectorProxy

def discover_strategies(candidates):
    strategies = []
    for name, emb_mod, det_mod in candidates:
        emb_fn = try_import_callable(emb_mod, "embedding")
        det_module_to_use = det_mod or emb_mod  # module path string
        # ensure detection exists in that module
        det_ok = False
        try:
            modtmp = importlib.import_module(det_module_to_use)
            if hasattr(modtmp, "detection") and callable(getattr(modtmp, "detection")):
                det_ok = True
        except Exception as e:
            det_ok = False

        if emb_fn is None:
            print(f"[STRAT] skipping '{name}': embedding not found.")
            continue
        if not det_ok:
            print(f"[STRAT] WARNING '{name}': detection not found in module '{det_module_to_use}'; skipping strategy.")
            continue

        # embedding: keep wrapper (called in main process)
        emb_wrapper = make_embedding_wrapper(emb_fn)
        # detection: create a picklable DetectorProxy instance (stores module path string)
        det_proxy = DetectorProxy(det_module_to_use)

        strategies.append({
            "name": name,
            "embedding_fn": emb_wrapper,
            "detection_fn": det_proxy,
            "emb_module": emb_mod,
            "det_module": det_module_to_use
        })
        print(f"[STRAT] Loaded strategy '{name}' (embedding: {emb_mod}, detection: {det_module_to_use})")
    return strategies


# ----------------------
# Main pipeline
# ----------------------
def main():
    # discover strategies
    strategies = discover_strategies(STRATEGIES_CANDIDATES)
    if not strategies:
        print("No strategies available. Exiting.")
        return

    # gather input images
    if not os.path.isdir(INPUT_DIR):
        print(f"Input dir not found: {INPUT_DIR}")
        return
    valid_exts = (".bmp", ".png", ".jpg", ".jpeg")
    image_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith(valid_exts)]
    if not image_files:
        print("No images found in input dir.")
        return

    # Prepare CSV
    csv_fields = ["strategy", "image", "watermarked_path", "best_attack_names", "best_attack_params", "best_wpsnr", "attack_success_bool", "notes"]
    csv_f = open(CSV_SUMMARY, "w", newline="")
    csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    csv_writer.writeheader()

    overall_start = time.time()

    # For each strategy: embed all images -> save in OUT_WM_ROOT/<strategy>/, then run brute-force for each image
    for strat in strategies:
        sname = strat["name"]
        print("\n" + "="*80)
        print(f"[STRATEGY] Running strategy: {sname}")
        print("="*80)
        out_wm_dir = os.path.join(OUT_WM_ROOT, sname)
        out_att_dir = os.path.join(OUT_ATT_ROOT, sname)
        os.makedirs(out_wm_dir, exist_ok=True)
        os.makedirs(out_att_dir, exist_ok=True)

        emb_fn = strat["embedding_fn"]
        det_fn = strat["detection_fn"]

        for img_name in image_files:
            input_path = os.path.join(INPUT_DIR, img_name)
            base_noext = os.path.splitext(img_name)[0]
            wm_out_name = f"{sname}_{img_name}"
            wm_out_path = os.path.join(out_wm_dir, wm_out_name)

            print(f"[{sname}] Embedding into image: {img_name} ...")
            try:
                # try call embedding(input1, input2)
                # assume watermark bits file is "ecorp.npy" in root if present, else pass None
                wm_bits = os.path.join(ROOT, "ecorp.npy")
                if not os.path.exists(wm_bits):
                    wm_bits = None
                Iw = emb_fn(input_path, wm_bits)
                if isinstance(Iw, tuple) or isinstance(Iw, list):
                    # some embedding variants might return (Iw, debug)
                    Iw = Iw[0]
                # ensure uint8 image
                Iw_u8 = Iw if Iw.dtype == 'uint8' else Iw.astype('uint8')
                cv2.imwrite(wm_out_path, Iw_u8)
            except Exception as e:
                print(f"[{sname}] ERROR embedding {img_name}: {e}")
                traceback.print_exc()
                csv_writer.writerow({
                    "strategy": sname,
                    "image": img_name,
                    "watermarked_path": "",
                    "best_attack_names": "",
                    "best_attack_params": "",
                    "best_wpsnr": 0.0,
                    "attack_success_bool": False,
                    "notes": f"embedding_failed:{e}"
                })
                continue

            # Run brute-force on this watermarked image
            print(f"[{sname}] Running brute-force attacks on {wm_out_path} ...")
            try:
                # find_best_attack(original_path, watermarked_path, adv_detection_func, max_workers)
                best_attack, best_wpsnr = bf.find_best_attack(
                    original_path=input_path,
                    watermarked_path=wm_out_path,
                    adv_detection_func=det_fn,
                    max_workers=os.cpu_count() or 4
                )
            except Exception as e:
                print(f"[{sname}] ERROR during brute-force for {img_name}: {e}")
                traceback.print_exc()
                csv_writer.writerow({
                    "strategy": sname,
                    "image": img_name,
                    "watermarked_path": wm_out_path,
                    "best_attack_names": "",
                    "best_attack_params": "",
                    "best_wpsnr": 0.0,
                    "attack_success_bool": False,
                    "notes": f"bruteforce_failed:{e}"
                })
                continue

            # If best attack found, generate final attacked image and save (in per-strategy attacked folder)
            attack_success_boolean = False
            attack_names = ""
            attack_params = ""
            if best_attack:
                attack_names = "+".join(best_attack.get("names", []))
                attack_params = str(best_attack.get("params", []))
                # final attacked image
                try:
                    final_att_img = bf.attacks(wm_out_path, best_attack["names"], best_attack["params"])
                    att_save_path = os.path.join(out_att_dir, f"{base_noext}_att_{sname}.bmp")
                    cv2.imwrite(att_save_path, final_att_img)
                except Exception as e:
                    print(f"[{sname}] Error generating final attacked image: {e}")
                    att_save_path = ""
                # success criteria is already enforced in find_best_attack (found==0 and wpsnr>=35)
                attack_success_boolean = True
            else:
                attack_names = ""
                attack_params = ""
                att_save_path = ""

            # log row
            csv_writer.writerow({
                "strategy": sname,
                "image": img_name,
                "watermarked_path": wm_out_path,
                "best_attack_names": attack_names,
                "best_attack_params": attack_params,
                "best_wpsnr": round(float(best_wpsnr), 4) if best_wpsnr else 0.0,
                "attack_success_bool": bool(attack_success_boolean),
                "notes": ""
            })

            # flush csv to disk progressively
            csv_f.flush()

        # per strategy end
        print(f"[{sname}] Done. Watermarked images stored in: {out_wm_dir}")
        print(f"[{sname}] Attacked images stored in: {out_att_dir}")

    csv_f.close()
    total_time = time.time() - overall_start
    print("\nAll strategies processed.")
    print(f"CSV summary saved to: {CSV_SUMMARY}")
    print(f"Total elapsed time: {round(total_time,1)} s")

    # Print a small summary table per strategy from the CSV
    try:
        from collections import defaultdict
        summary = defaultdict(lambda: {"images":0, "successes":0})
        with open(CSV_SUMMARY, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                s = r["strategy"]
                summary[s]["images"] += 1
                if r["attack_success_bool"].lower() in ("true","1","yes"):
                    summary[s]["successes"] += 1
        print("\nSummary (strategy : successes / images):")
        for s, v in summary.items():
            print(f"  {s} : {v['successes']} / {v['images']}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
