#!/usr/bin/env python3
"""
Preprocess rsud20k_person2000 by resizing images to 640x640 and
creating a clean YOLO-ready directory with data.yaml.

Input (already filtered to person-containing images):
  rsud20k_person2000/
    images/{train,val,test}
    labels/{train,val,test}

Output:
  rsud20k_person2000_resized/
    images/{train,val,test}
    labels/{train,val,test}
  data/rsud20k_person2000_resized.yaml
"""

from pathlib import Path
from shutil import copy2, rmtree
from PIL import Image
from tqdm import tqdm


SRC_ROOT = Path("rsud20k_person2000")
DST_ROOT = Path("rsud20k_person2000_resized")
SIZE = 640

CLASS_NAMES = [
    "person",
    "rickshaw",
    "rickshaw van",
    "auto rickshaw",
    "truck",
    "pickup truck",
    "private car",
    "motorcycle",
    "bicycle",
    "bus",
    "micro bus",
    "covered van",
    "human hauler",
]


def resize_and_copy(src_img: Path, src_label: Path, dst_img: Path, dst_label: Path):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(src_img).convert("RGB")
    img = img.resize((SIZE, SIZE), Image.BILINEAR)
    img.save(dst_img, quality=95)

    if src_label.exists():
        copy2(src_label, dst_label)


def process_split(split: str):
    src_img_dir = SRC_ROOT / "images" / split
    src_lbl_dir = SRC_ROOT / "labels" / split
    dst_img_dir = DST_ROOT / "images" / split
    dst_lbl_dir = DST_ROOT / "labels" / split

    imgs = sorted(src_img_dir.glob("*.jpg"))
    for img in tqdm(imgs, desc=f"{split}"):
        lbl = src_lbl_dir / f"{img.stem}.txt"
        dst_img = dst_img_dir / img.name
        dst_lbl = dst_lbl_dir / lbl.name
        resize_and_copy(img, lbl, dst_img, dst_lbl)


def write_yaml():
    yaml_path = Path("data/rsud20k_person2000_resized.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""path: {DST_ROOT}

train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path.write_text(content)
    print(f"data.yaml written to {yaml_path}")


def main():
    if DST_ROOT.exists():
        rmtree(DST_ROOT)
    for split in ["train", "val", "test"]:
        process_split(split)
    write_yaml()
    print(f"Done. YOLO-ready dataset at {DST_ROOT}")


if __name__ == "__main__":
    main()

