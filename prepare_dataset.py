"""Convert the Kaggle ODIR-5K dataset into the folder layout train.py expects.

ODIR-5K ships one CSV row per patient, with 0/1 label columns N, D, G, C, A, H, M, O and
two image file names (Left-Fundus, Right-Fundus). This script:

  * keeps only patients that have EXACTLY ONE label (train.py is single-label / softmax),
  * copies both eye images into  <out>/<split>/<Class name>/,
  * splits train/test BY PATIENT so the two eyes of one person never end up in both sets.

Usage (adjust the paths to where you extracted the Kaggle download):
    python prepare_dataset.py --csv path/to/full_df.csv --images-dir path/to/images --out data

Dataset: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
"""

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd

# ODIR column -> folder name. Folder names must match DISEASE_LABELS in train.py / app.py.
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


def find_image(images_dir: Path, name: str):
    p = images_dir / name
    if p.is_file():
        return p
    matches = list(images_dir.rglob(name))  # tolerate sub-folders
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="ODIR CSV (e.g. full_df.csv / data.csv).")
    ap.add_argument("--images-dir", type=Path, required=True, help="Folder containing the fundus images.")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing_cols = [c for c in [*CODE_TO_CLASS, "Left-Fundus", "Right-Fundus"] if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"CSV is missing expected columns {missing_cols}. Found: {list(df.columns)}")

    codes = list(CODE_TO_CLASS)
    df = df[df[codes].sum(axis=1) == 1].copy()
    print(f"Patients with exactly one label: {len(df)}")

    rng = random.Random(args.seed)
    patient_idx = list(df.index)
    rng.shuffle(patient_idx)
    n_test = int(len(patient_idx) * args.test_fraction)
    test_set = set(patient_idx[:n_test])

    counts, not_found = {}, 0
    for idx, row in df.iterrows():
        cls = CODE_TO_CLASS[next(c for c in codes if row[c] == 1)]
        split = "test" if idx in test_set else "train"
        for col in ("Left-Fundus", "Right-Fundus"):
            src = find_image(args.images_dir, str(row[col]))
            if src is None:
                not_found += 1
                continue
            dst_dir = args.out / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)
            counts[(split, cls)] = counts.get((split, cls), 0) + 1

    print(f"\nImages not found: {not_found}")
    for split in ("train", "test"):
        print(f"\n{split}:")
        for cls in CODE_TO_CLASS.values():
            print(f"  {cls:22s} {counts.get((split, cls), 0)}")
    print(f"\nDone. Now run:  python train.py --data-dir {args.out}")


if __name__ == "__main__":
    main()
