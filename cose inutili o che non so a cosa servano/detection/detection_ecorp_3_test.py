#!/usr/bin/env python3
"""
check_attacked_detection.py

Scorri una cartella di immagini attaccate e, per ciascuna, prova a trovare:
 - l'immagine originale in 'images/' (matching by suffix/substring)
 - l'immagine watermarked in 'watermarked_images/' (typical name: watermarked_<original>)

Poi chiama la funzione detection(original, watermarked, attacked) che deve restituire:
    dec, wpsnr

Produce:
 - CSV report 'attacked_detection_report.csv'
 - stampa sommario a terminale

Adatta i nomi delle cartelle se nella tua struttura sono diversi.
"""

import os
import csv
import sys

from detection.detection_ecorp_3 import detection  # <-- USO COME RICHIESTO
from wpsnr import wpsnr

# Config
ATTACKED_DIR = "attacked_images"           # cartella con immagini attaccate (modifica se serve)
WATERMARKED_DIR = "watermarked_images"     # cartella dove sono salvate le watermarked
ORIGINALS_DIR = "images"                   # cartella con le originali
OUTPUT_CSV = "attacked_detection_report.csv"

# try import detection
try:
    from detection.detection_ecorp_3 import detection
except Exception as e:
    print("Error: cannot import detection.detection_ecorp_3.detection().")
    print("Make sure detection module exists and is in PYTHONPATH.")
    print("Import error:", e)
    sys.exit(1)

def find_original_filename(attacked_fname, originals_list):
    """
    Try to find a candidate original filename from attacked filename.
    Strategy: choose original name that is a suffix of attacked_fname (case-insensitive),
    or a substring. Returns filename or None.
    """
    low_att = attacked_fname.lower()
    # 1) suffix match (best)
    for orig in originals_list:
        if low_att.endswith(orig.lower()):
            return orig
    # 2) substring match
    for orig in originals_list:
        if orig.lower() in low_att:
            return orig
    # 3) fallback: if attacked contains "watermarked_" remove prefix patterns and try
    for orig in originals_list:
        if orig.lower().replace(" ", "_") in low_att:
            return orig
    return None

def find_watermarked_for_original(orig_fname, watermarked_list):
    """
    Find watermarked filename corresponding to original, typical naming:
      watermarked_<orig_fname>
    or contains orig_fname as suffix/substring.
    """
    # exact watermarked_<orig>
    candidate = f"watermarked_{orig_fname}"
    if candidate in watermarked_list:
        return candidate
    # suffix match
    for wm in watermarked_list:
        if wm.lower().endswith(orig_fname.lower()):
            return wm
    # substring match
    for wm in watermarked_list:
        if orig_fname.lower() in wm.lower():
            return wm
    return None

def main():
    if not os.path.isdir(ATTACKED_DIR):
        print(f"Attacked directory not found: {ATTACKED_DIR}")
        return
    if not os.path.isdir(ORIGINALS_DIR):
        print(f"Originals directory not found: {ORIGINALS_DIR}")
        return
    if not os.path.isdir(WATERMARKED_DIR):
        print(f"Watermarked directory not found: {WATERMARKED_DIR}")
        return

    attacked_files = sorted([f for f in os.listdir(ATTACKED_DIR) if f.lower().endswith(('.bmp','.png','.jpg','.jpeg','.tiff'))])
    originals = sorted([f for f in os.listdir(ORIGINALS_DIR) if f.lower().endswith(('.bmp','.png','.jpg','.jpeg','.tiff'))])
    watermarked = sorted([f for f in os.listdir(WATERMARKED_DIR) if f.lower().endswith(('.bmp','.png','.jpg','.jpeg','.tiff'))])

    if not attacked_files:
        print("No attacked images found in", ATTACKED_DIR)
        return

    # Prepare CSV
    csv_rows = []
    csv_header = ["attacked_file", "original_file", "watermarked_file", "dec", "wpsnr", "note"]

    detected_count = 0
    total = 0

    for af in attacked_files:
        total += 1
        attacked_path = os.path.join(ATTACKED_DIR, af)
        note = ""
        orig_match = find_original_filename(af, originals)
        if orig_match is None:
            # If no original found, we will try to infer original from watermarked names by removing prefixes
            # Try to infer original by searching originals that appear in watermarked filenames that are substrings of af
            inferred = None
            for wm in watermarked:
                if wm.lower() in af.lower() or af.lower() in wm.lower():
                    # attempt to remove "watermarked_" prefix
                    candidate = wm
                    if candidate.startswith("watermarked_"):
                        candidate = candidate[len("watermarked_"):]
                    if candidate in originals:
                        inferred = candidate
                        break
            if inferred:
                orig_match = inferred
                note += "original_inferred_from_watermarked; "
            else:
                note += "original_not_found; "

        wm_match = None
        if orig_match:
            wm_match = find_watermarked_for_original(orig_match, watermarked)
            if wm_match is None:
                # try to deduce watermarked name variants from attacked filename
                # look for any watermarked file that contains attacked filename (use substring)
                for wm in watermarked:
                    if af.lower().endswith(wm.lower()) or wm.lower() in af.lower() or af.lower() in wm.lower():
                        wm_match = wm
                        note += "watermarked_guess_from_attacked; "
                        break

        # Fall back: if no watermarked found, try to pick any watermarked file that contains original name as substring
        if wm_match is None and orig_match:
            for wm in watermarked:
                if orig_match.lower() in wm.lower():
                    wm_match = wm
                    note += "watermarked_fallback_by_original; "
                    break

        # As last resort, if still None and only one watermarked file exists, use that (but note it)
        if wm_match is None and len(watermarked) == 1:
            wm_match = watermarked[0]
            note += "watermarked_single_file_used; "

        # if still None, set note and skip detection
        if wm_match is None or orig_match is None:
            note += "skipped_detection_missing_files"
            csv_rows.append([af, orig_match if orig_match else "", wm_match if wm_match else "", "", "", note.strip()])
            print(f"[{af}] SKIPPED: {note.strip()}")
            continue

        orig_path = os.path.join(ORIGINALS_DIR, orig_match)
        wm_path = os.path.join(WATERMARKED_DIR, wm_match)

        # Call detection
        try:
            dec, wps = detection(orig_path, wm_path, attacked_path)
            note += "ok"
        except Exception as e:
            dec = ""
            wps = ""
            note += f"detection_exception:{e!s}"

        if dec == 1:
            detected_count += 1

        csv_rows.append([af, orig_match, wm_match, dec, wps, note.strip()])
        print(f"[{af}] -> original: {orig_match}, watermarked: {wm_match}, dec={dec}, wpsnr={wps}")

    # Save CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)

    print()
    print("Summary:")
    print(f"  Total attacked files scanned: {total}")
    print(f"  Detected (dec==1): {detected_count}")
    print(f"  Not-detected or skipped: {total - detected_count}")
    print(f"  CSV report saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
