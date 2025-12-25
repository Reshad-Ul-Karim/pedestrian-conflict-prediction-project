#!/usr/bin/env python3
"""
Create a 2000-image YOLO dataset containing the 'person' class from RSUD20K.

Output structure:
  rsud20k_person2000/
    images/{train,val,test}
    labels/{train,val,test}

Split: 70% train, 15% val, 15% test (1400 / 300 / 300)
Only images that contain at least one person (class id 0) are included.
"""

import random
from pathlib import Path
from shutil import copy2
from typing import List, Tuple


SRC_ROOT = Path("rsud20k")
DST_ROOT = Path("rsud20k_person2000")
TOTAL = 2000
SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
PERSON_CLASS_ID = "0"  # person is the first class in classes.txt


def find_person_images() -> List[Tuple[Path, Path]]:
    """Return list of (image_path, label_path) that contain person."""
    pairs = []
    for split in ["train", "val", "test"]:
        label_dir = SRC_ROOT / "labels" / split
        img_dir = SRC_ROOT / "images" / split
        for lbl in label_dir.glob("*.txt"):
            try:
                txt = lbl.read_text().strip()
            except Exception:
                continue
            if not txt:
                continue
            # Check if any line starts with person class id
            if any(line.startswith(f"{PERSON_CLASS_ID} ") or line == PERSON_CLASS_ID for line in txt.splitlines()):
                img = img_dir / f"{lbl.stem}.jpg"
                if img.exists():
                    pairs.append((img, lbl))
    return pairs


def split_pairs(pairs: List[Tuple[Path, Path]]) -> dict:
    """Split into train/val/test according to SPLIT proportions."""
    n = len(pairs)
    n_train = int(TOTAL * SPLIT["train"])
    n_val = int(TOTAL * SPLIT["val"])
    n_test = TOTAL - n_train - n_val
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val : n_train + n_val + n_test],
    }


def copy_subset(subset: List[Tuple[Path, Path]], split: str):
    """Copy images and labels to destination split."""
    dst_img_dir = DST_ROOT / "images" / split
    dst_lbl_dir = DST_ROOT / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img, lbl in subset:
        copy2(img, dst_img_dir / img.name)
        copy2(lbl, dst_lbl_dir / lbl.name)


def main():
    pairs = find_person_images()
    if len(pairs) < TOTAL:
        raise RuntimeError(f"Not enough person images. Found {len(pairs)}, need {TOTAL}.")

    random.seed(42)
    random.shuffle(pairs)
    pairs = pairs[:TOTAL]

    splits = split_pairs(pairs)

    # Clear destination if exists
    if DST_ROOT.exists():
        import shutil

        shutil.rmtree(DST_ROOT)

    for split, subset in splits.items():
        copy_subset(subset, split)
        print(f"{split}: {len(subset)}")

    print(f"\nDone. Dataset created at: {DST_ROOT}")


if __name__ == "__main__":
    main()

