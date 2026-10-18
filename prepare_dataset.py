"""Convert the Kaggle ODIR-5K dataset into the folder layout train.py expects.

Supports the Kaggle file `full_df.csv` (one row per EYE image, columns `filename` and
`labels`, e.g. "['N']") and falls back to the original patient-level format
(columns Left-Fundus, Right-Fundus and 0/1 columns N, D, G, C, A, H, M, O).

What it does:
  * keeps only images that have exactly one label (train.py is single-label / softmax),
  * copies them into  <out>/<split>/<Class name>/,
  * splits train/test BY PATIENT (ID column) so both eyes of one person never end up in
    different splits.

Usage:
    python prepare_dataset.py --csv path/to/full_df.csv --images-dir path/to/preprocessed_images --out data

Dataset: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
"""

import argparse
import ast
import random
import shutil
from pathlib import Path

import pandas as pd

# ODIR code -> folder name. Folder names must match DISEASE_LABELS in train.py / app.py.
# Note: ODIR's "D" means diabetes-related retinal disease; it is used here as
# "Diabetic Retinopathy" to match the labels in app.py.
CODE_TO_CLASS = {
    "N": "Normal",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other",
}
CODES = list(CODE_TO_CLASS)


def parse_labels(value):
    """'['N']' -> ['N'] (also accepts a plain list)."""
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        parsed = ast.literal_eval(str(value))
        return list(parsed) if isinstance(parsed, (list, tuple)) else [str(parsed)]
    except (ValueError, SyntaxError):
        return []


def build_records(df):
    """Return a list of (patient_id, image_filename, class_name) with exactly one label."""
    records = []
    if {"filename", "labels"}.issubset(df.columns):
        print("Format: one row per eye image (filename + labels).")
        for i, row in df.iterrows():
            labs = [c for c in parse_labels(row["labels"]) if c in CODE_TO_CLASS]
            if len(labs) == 1:
                records.append((row["ID"] if "ID" in df.columns else i, str(row["filename"]), CODE_TO_CLASS[labs[0]]))
    elif {"Left-Fundus", "Right-Fundus", *CODES}.issubset(df.columns):
        print("Format: one row per patient (Left-Fundus / Right-Fundus + N..O flags).")
        for i, row in df.iterrows():
            flags = [c for c in CODES if row[c] == 1]
            if len(flags) == 1:
                pid = row["ID"] if "ID" in df.columns else i
                records.append((pid, str(row["Left-Fundus"]), CODE_TO_CLASS[flags[0]]))
                records.append((pid, str(row["Right-Fundus"]), CODE_TO_CLASS[flags[0]]))
    else:
        raise SystemExit(f"Unrecognised CSV format. Columns found: {list(df.columns)}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    records = build_records(df)
    print(f"Images with exactly one label: {len(records)}")

    patients = sorted({r[0] for r in records}, key=str)
    random.Random(args.seed).shuffle(patients)
    test_patients = set(patients[: int(len(patients) * args.test_fraction)])

    counts, not_found = {}, 0
    for pid, fname, cls in records:
        src = args.images_dir / fname
        if not src.is_file():
            not_found += 1
            continue
        split = "test" if pid in test_patients else "train"
        dst_dir = args.out / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        counts[(split, cls)] = counts.get((split, cls), 0) + 1

    print(f"\nImages not found: {not_found}")
    for split in ("train", "test"):
        print(f"\n{split}:")
        for cls in CODE_TO_CLASS.values():
            print(f"  {cls:22s} {counts.get((split, cls), 0)}")
    if not_found == len(records):
        raise SystemExit("No images were found - check --images-dir.")
    print(f"\nDone. Now run:  python train.py --data-dir {args.out}")


if __name__ == "__main__":
    main()
