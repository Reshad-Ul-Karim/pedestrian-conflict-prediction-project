#!/usr/bin/env python3
"""
Resize RSUD20K images to 640x640 for faster YOLO training.

Creates a smaller 500-image train subset plus full val set:
- Input root:  rsud20k/
- Output root: rsud20k_640/
"""

from pathlib import Path
from shutil import copy2

from PIL import Image
from tqdm import tqdm


def resize_and_copy(
    src_img: Path,
    src_label_root: Path,
    dst_img_root: Path,
    dst_label_root: Path,
    size: int = 640,
):
    """Resize image to size x size and copy label file."""
    dst_img_root.mkdir(parents=True, exist_ok=True)
    dst_label_root.mkdir(parents=True, exist_ok=True)

    # Image
    img = Image.open(src_img).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    dst_img = dst_img_root / src_img.name
    img.save(dst_img, quality=95)

    # Label (YOLO txt, normalized -> unchanged)
    src_label = src_label_root / (src_img.stem + ".txt")
    if src_label.exists():
        dst_label = dst_label_root / src_label.name
        copy2(src_label, dst_label)


def build_rsud20k_640(
    src_root: Path = Path("rsud20k"),
    dst_root: Path = Path("rsud20k_640"),
    train_limit: int = 500,
    size: int = 640,
):
    # Train subset (500 images)
    src_train_imgs = sorted((src_root / "images" / "train").glob("*.jpg"))
    src_train_labels = src_root / "labels" / "train"
    dst_train_imgs = dst_root / "images" / "train"
    dst_train_labels = dst_root / "labels" / "train"

    subset = src_train_imgs[:train_limit]
    print(f"Building train subset: {len(subset)} images -> {dst_train_imgs}")
    for img_path in tqdm(subset, desc="Train 500 resize"):
        resize_and_copy(
            img_path,
            src_train_labels,
            dst_train_imgs,
            dst_train_labels,
            size=size,
        )

    # Full val set
    src_val_imgs = sorted((src_root / "images" / "val").glob("*.jpg"))
    src_val_labels = src_root / "labels" / "val"
    dst_val_imgs = dst_root / "images" / "val"
    dst_val_labels = dst_root / "labels" / "val"

    print(f"Building val set: {len(src_val_imgs)} images -> {dst_val_imgs}")
    for img_path in tqdm(src_val_imgs, desc="Val resize"):
        resize_and_copy(
            img_path,
            src_val_labels,
            dst_val_imgs,
            dst_val_labels,
            size=size,
        )

    print("Done. You can now point YOLO to rsud20k_640 in your data YAML.")


if __name__ == "__main__":
    build_rsud20k_640()



