import os
import sys
import argparse
import importlib.util
from pathlib import Path
from shutil import copyfile
import traceback

def load_module_from_path(name, path):
    """Carica dinamicamente un modulo da file .py o .pyc"""
    import importlib.util, importlib.machinery
    if path.endswith(".py"):
        spec = importlib.util.spec_from_file_location(name, path)
    else:
        loader = importlib.machinery.SourcelessFileLoader(name, path)
        spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    parser = argparse.ArgumentParser(description="Versione semplificata BFA runner")
    parser.add_argument("--detection", required=True, help="Percorso al file detection.py o .pyc")
    parser.add_argument("--bfa", required=True, help="Percorso al file attack_brute_force.py")
    parser.add_argument("--originals", required=True, help="Cartella immagini originali")
    parser.add_argument("--watermarked", required=True, help="Cartella immagini watermarked")
    parser.add_argument("--attacked", required=True, help="Cartella output attacchi")
    parser.add_argument("--removed", required=True, help="Cartella per immagini dove watermark è rimosso")
    parser.add_argument("--group", default="manual", help="Nome gruppo o test (per filename output)")
    args = parser.parse_args()

    # Caricamento moduli
    print(f"[LOAD] detection: {args.detection}")
    det_mod = load_module_from_path("detection_module_manual", args.detection)
    if not hasattr(det_mod, "detection"):
        raise RuntimeError("Il modulo detection non contiene 'detection()'")
    detection_fn = det_mod.detection

    print(f"[LOAD] BFA: {args.bfa}")
    bfa_mod = load_module_from_path("attack_brute_force", args.bfa)
    if not hasattr(bfa_mod, "find_best_attack"):
        raise RuntimeError("Il modulo BFA non contiene 'find_best_attack()'")

    originals_dir = Path(args.originals)
    wm_dir = Path(args.watermarked)
    attacked_dir = Path(args.attacked)
    removed_dir = Path(args.removed)

    attacked_dir.mkdir(exist_ok=True, parents=True)
    removed_dir.mkdir(exist_ok=True, parents=True)

    # Processa ogni immagine watermarked
    wm_images = [f for f in wm_dir.iterdir() if f.suffix.lower() in [".bmp", ".png", ".jpg", ".jpeg"]]
    print(f"\n[INFO] Trovate {len(wm_images)} immagini watermarked.\n")

    for wm_path in wm_images:
        name = wm_path.name
        original_path = originals_dir / name
        if not original_path.exists():
            print(f"⚠️  Originale mancante per {name}, skip.")
            continue

        print(f"[ATTACK] {name}")
        try:
            best_attack, best_wpsnr = bfa_mod.find_best_attack(
                original_path=str(original_path),
                watermarked_path=str(wm_path),
                adv_detection_func=detection_fn,
                max_workers=1  # sicuro per tutti i casi
            )
        except Exception as e:
            print(f"❌ Errore durante attacco {name}: {e}")
            traceback.print_exc()
            continue

        if best_attack is None:
            print("  Nessun attacco efficace trovato (criterio WPSNR non soddisfatto)")
            continue

        # Salva immagine attaccata
        attacked_image = bfa_mod.attacks(
            str(wm_path),
            [best_attack[0]],  # nome attacco
            [best_attack[1]]   # parametri
        )

        out_path = attacked_dir / name
        attacked_image.save(out_path)
        print(f"  ✅ Salvata immagine attaccata in {out_path}")

        # Verifica se watermark è rimosso
        detection_ok = detection_fn(str(original_path), str(wm_path), str(out_path))
        if not detection_ok:
            removed_name = f"{wm_path.stem}_{args.group}_{round(best_wpsnr,2)}_{best_attack[0]}.bmp"
            removed_path = removed_dir / removed_name
            copyfile(out_path, removed_path)
            print(f"  💥 Watermark rimosso! Copiata in {removed_path}")

    print("\n=== COMPLETATO ===")

if __name__ == "__main__":
    main()
