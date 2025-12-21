#!/usr/bin/env python3
"""
Partially download and prepare RSUD20K dataset for YOLO training
Downloads a subset of the dataset and converts annotations to YOLO format
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm
import cv2
from PIL import Image


class RSUD20KDownloader:
    """Download and prepare RSUD20K dataset"""
    
    def __init__(
        self,
        data_dir: str = "data/rsud20k",
        download_percentage: float = 0.1,  # Download 10% of dataset
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        """
        Args:
            data_dir: Directory to store the dataset
            download_percentage: Percentage of dataset to download (0.0 to 1.0)
            train_split: Training set proportion
            val_split: Validation set proportion
            test_split: Test set proportion
        """
        self.data_dir = Path(data_dir)
        self.download_percentage = download_percentage
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # RSUD20K GitHub repository
        self.repo_url = "https://github.com/hasibzunair/RSUD20K"
        self.raw_base_url = "https://raw.githubusercontent.com/hasibzunair/RSUD20K/main"
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "images").mkdir(exist_ok=True)
        (self.data_dir / "annotations").mkdir(exist_ok=True)
        (self.data_dir / "labels").mkdir(exist_ok=True)
        
        # Class mapping for RSUD20K (13 classes)
        self.class_names = [
            "person", "bicycle", "motorcycle", "car", "bus", "truck",
            "rickshaw", "CNG", "van", "pickup", "ambulance", "fire_truck", "police"
        ]
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
    def download_file(self, url: str, dest_path: Path) -> bool:
        """Download a file from URL"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def get_file_list_from_github(self) -> Tuple[List[str], List[str]]:
        """
        Get list of files from RSUD20K repository
        Returns: (image_files, annotation_files)
        """
        # Since we can't easily list GitHub files via API without authentication,
        # we'll use a known structure or download a manifest file
        # For now, we'll create a sample structure
        
        print("Note: RSUD20K dataset structure may vary.")
        print("This script will attempt to download from common paths.")
        print("You may need to manually specify file lists if the structure differs.\n")
        
        # Sample file structure (adjust based on actual RSUD20K structure)
        # In practice, you might need to clone the repo or use their API
        image_files = []
        annotation_files = []
        
        return image_files, annotation_files
    
    def download_partial_dataset(self, max_images: int = 500):
        """
        Download a partial dataset
        For RSUD20K, we'll use a simplified approach:
        1. Try to clone a subset using git sparse-checkout
        2. Or download specific files if URLs are known
        """
        print(f"Downloading partial RSUD20K dataset (target: ~{max_images} images)")
        print("=" * 60)
        
        # Method 1: Try git sparse-checkout (recommended)
        repo_path = self.data_dir / "raw"
        if not (repo_path / ".git").exists():
            print("\nAttempting to clone RSUD20K repository with sparse checkout...")
            print("This will download only a subset of files.")
            
            try:
                import subprocess
                
                # Initialize git repo
                subprocess.run(
                    ["git", "init"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True
                )
                
                # Configure sparse checkout
                subprocess.run(
                    ["git", "config", "core.sparseCheckout", "true"],
                    cwd=repo_path,
                    check=True
                )
                
                # Create sparse-checkout file (download only images and annotations)
                sparse_file = repo_path / ".git" / "info" / "sparse-checkout"
                sparse_file.parent.mkdir(parents=True, exist_ok=True)
                with open(sparse_file, 'w') as f:
                    f.write("images/*\n")
                    f.write("annotations/*\n")
                    f.write("*.json\n")
                    f.write("*.txt\n")
                
                # Add remote
                subprocess.run(
                    ["git", "remote", "add", "origin", self.repo_url + ".git"],
                    cwd=repo_path,
                    check=False  # May already exist
                )
                
                print("Cloning repository (this may take a while)...")
                result = subprocess.run(
                    ["git", "pull", "origin", "main"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print("✓ Repository cloned successfully")
                else:
                    print(f"⚠ Git clone failed: {result.stderr}")
                    print("Falling back to manual download method...")
                    return self._download_manual_sample(max_images)
                    
            except Exception as e:
                print(f"⚠ Git method failed: {e}")
                print("Falling back to manual download method...")
                return self._download_manual_sample(max_images)
        else:
            print("Repository already exists, using existing files...")
        
        # Process downloaded files
        return self._process_downloaded_files(repo_path, max_images)
    
    def _download_manual_sample(self, max_images: int):
        """Download a small sample manually (for testing)"""
        print("\nUsing manual download method...")
        print("Note: For full dataset, please clone RSUD20K repository manually:")
        print(f"  git clone {self.repo_url}.git {self.data_dir / 'raw'}")
        print("\nCreating sample structure for testing...")
        
        # Create a minimal sample structure
        sample_dir = self.data_dir / "raw" / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a README with download instructions
        readme_path = self.data_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(readme_path, 'w') as f:
            f.write("RSUD20K Dataset Download Instructions\n")
            f.write("=" * 60 + "\n\n")
            f.write("To download the full RSUD20K dataset:\n\n")
            f.write("1. Clone the repository:\n")
            f.write(f"   git clone {self.repo_url}.git data/rsud20k/raw\n\n")
            f.write("2. Or download from the official source:\n")
            f.write(f"   Visit: {self.repo_url}\n\n")
            f.write("3. For partial download, use git sparse-checkout:\n")
            f.write("   cd data/rsud20k/raw\n")
            f.write("   git init\n")
            f.write("   git config core.sparseCheckout true\n")
            f.write("   echo 'images/*' >> .git/info/sparse-checkout\n")
            f.write("   echo 'annotations/*' >> .git/info/sparse-checkout\n")
            f.write("   git remote add origin {self.repo_url}.git\n")
            f.write("   git pull origin main\n\n")
            f.write("After downloading, run this script again to process the data.\n")
        
        print(f"✓ Created download instructions at: {readme_path}")
        return False
    
    def _process_downloaded_files(self, repo_path: Path, max_images: int) -> bool:
        """Process downloaded files and convert to YOLO format"""
        print("\nProcessing downloaded files...")
        
        # Find image and annotation files
        images_dir = repo_path / "images"
        annotations_dir = repo_path / "annotations"
        
        if not images_dir.exists():
            print(f"⚠ Images directory not found at {images_dir}")
            return False
        
        # Get list of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
        
        if not image_files:
            print("⚠ No image files found")
            return False
        
        # Limit to max_images
        if len(image_files) > max_images:
            print(f"Selecting {max_images} random images from {len(image_files)} available...")
            image_files = random.sample(image_files, max_images)
        
        print(f"Processing {len(image_files)} images...")
        
        # Process each image
        processed = 0
        for img_path in tqdm(image_files, desc="Converting to YOLO format"):
            try:
                # Copy image
                dest_img = self.data_dir / "images" / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Find corresponding annotation
                annotation_path = self._find_annotation(img_path, annotations_dir)
                
                if annotation_path and annotation_path.exists():
                    # Convert annotation to YOLO format
                    self._convert_to_yolo(annotation_path, dest_img, img_path.stem)
                    processed += 1
                else:
                    print(f"⚠ No annotation found for {img_path.name}")
                    
            except Exception as e:
                print(f"⚠ Error processing {img_path.name}: {e}")
        
        print(f"\n✓ Processed {processed} images with annotations")
        return processed > 0
    
    def _find_annotation(self, img_path: Path, annotations_dir: Path) -> Path:
        """Find corresponding annotation file"""
        # Try different annotation formats
        base_name = img_path.stem
        
        # Try JSON format
        json_path = annotations_dir / f"{base_name}.json"
        if json_path.exists():
            return json_path
        
        # Try XML format (PASCAL VOC)
        xml_path = annotations_dir / f"{base_name}.xml"
        if xml_path.exists():
            return xml_path
        
        # Try TXT format (might already be YOLO)
        txt_path = annotations_dir / f"{base_name}.txt"
        if txt_path.exists():
            return txt_path
        
        return None
    
    def _convert_to_yolo(self, annotation_path: Path, image_path: Path, base_name: str):
        """Convert annotation to YOLO format"""
        # Read image to get dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            return
        img_h, img_w = img.shape[:2]
        
        # Determine annotation format and convert
        if annotation_path.suffix == '.json':
            self._convert_json_to_yolo(annotation_path, image_path, base_name, img_w, img_h)
        elif annotation_path.suffix == '.xml':
            self._convert_xml_to_yolo(annotation_path, image_path, base_name, img_w, img_h)
        elif annotation_path.suffix == '.txt':
            # Might already be YOLO format, just copy
            dest_label = self.data_dir / "labels" / f"{base_name}.txt"
            shutil.copy2(annotation_path, dest_label)
    
    def _convert_json_to_yolo(self, json_path: Path, image_path: Path, base_name: str, img_w: int, img_h: int):
        """Convert JSON annotation to YOLO format"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # YOLO format: class_id x_center y_center width height (normalized)
            yolo_lines = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                annotations = data
            elif isinstance(data, dict):
                annotations = data.get('annotations', data.get('objects', []))
            else:
                annotations = []
            
            for ann in annotations:
                # Extract class name
                class_name = ann.get('class', ann.get('category', ann.get('name', '')))
                if class_name not in self.class_to_id:
                    continue
                
                class_id = self.class_to_id[class_name]
                
                # Extract bbox (handle different formats)
                bbox = ann.get('bbox', ann.get('bounding_box', ann.get('box', [])))
                if not bbox or len(bbox) < 4:
                    continue
                
                # Convert to YOLO format
                if len(bbox) == 4:
                    x_min, y_min, width, height = bbox
                    x_max = x_min + width
                    y_max = y_min + height
                elif len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                    # Assume [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = bbox
                    width = x_max - x_min
                    height = y_max - y_min
                else:
                    continue
                
                # Normalize coordinates
                x_center = (x_min + width / 2) / img_w
                y_center = (y_min + height / 2) / img_h
                width_norm = width / img_w
                height_norm = height / img_h
                
                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width_norm = max(0, min(1, width_norm))
                height_norm = max(0, min(1, height_norm))
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
            
            # Write YOLO format file
            if yolo_lines:
                label_path = self.data_dir / "labels" / f"{base_name}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines) + '\n')
                    
        except Exception as e:
            print(f"⚠ Error converting {json_path}: {e}")
    
    def _convert_xml_to_yolo(self, xml_path: Path, image_path: Path, base_name: str, img_w: int, img_h: int):
        """Convert XML (PASCAL VOC) annotation to YOLO format"""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            yolo_lines = []
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in self.class_to_id:
                    continue
                
                class_id = self.class_to_id[class_name]
                
                bbox = obj.find('bndbox')
                x_min = float(bbox.find('xmin').text)
                y_min = float(bbox.find('ymin').text)
                x_max = float(bbox.find('xmax').text)
                y_max = float(bbox.find('ymax').text)
                
                # Normalize
                width = x_max - x_min
                height = y_max - y_min
                x_center = (x_min + width / 2) / img_w
                y_center = (y_min + height / 2) / img_h
                width_norm = width / img_w
                height_norm = height / img_h
                
                # Clamp
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width_norm = max(0, min(1, width_norm))
                height_norm = max(0, min(1, height_norm))
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
            
            if yolo_lines:
                label_path = self.data_dir / "labels" / f"{base_name}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines) + '\n')
                    
        except Exception as e:
            print(f"⚠ Error converting {xml_path}: {e}")
    
    def create_splits(self):
        """Create train/val/test splits"""
        print("\nCreating dataset splits...")
        
        # Get all image files
        images_dir = self.data_dir / "images"
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        image_files = [f.stem for f in image_files]  # Get base names
        
        # Filter to only include images with labels
        valid_images = []
        labels_dir = self.data_dir / "labels"
        for img_name in image_files:
            label_path = labels_dir / f"{img_name}.txt"
            if label_path.exists():
                valid_images.append(img_name)
        
        if not valid_images:
            print("⚠ No valid image-label pairs found")
            return
        
        # Shuffle
        random.shuffle(valid_images)
        
        # Split
        n_total = len(valid_images)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        n_test = n_total - n_train - n_val
        
        train_images = valid_images[:n_train]
        val_images = valid_images[n_train:n_train + n_val]
        test_images = valid_images[n_train + n_val:]
        
        print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
        # Create split directories
        for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
            split_img_dir = self.data_dir / "images" / split_name
            split_label_dir = self.data_dir / "labels" / split_name
            split_img_dir.mkdir(exist_ok=True)
            split_label_dir.mkdir(exist_ok=True)
            
            for img_name in split_images:
                # Move images
                src_img = self.data_dir / "images" / f"{img_name}.jpg"
                if not src_img.exists():
                    src_img = self.data_dir / "images" / f"{img_name}.png"
                if src_img.exists():
                    shutil.move(str(src_img), str(split_img_dir / src_img.name))
                
                # Move labels
                src_label = self.data_dir / "labels" / f"{img_name}.txt"
                if src_label.exists():
                    shutil.move(str(src_label), str(split_label_dir / src_label.name))
        
        print("✓ Dataset splits created")
    
    def create_yolo_config(self):
        """Create YOLO dataset configuration file"""
        config_path = self.data_dir / "dataset.yaml"
        
        config = {
            'path': str(self.data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ YOLO config created: {config_path}")
        return config_path


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and prepare RSUD20K dataset")
    parser.add_argument("--data-dir", type=str, default="data/rsud20k",
                       help="Directory to store dataset")
    parser.add_argument("--max-images", type=int, default=500,
                       help="Maximum number of images to download")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, only process existing files")
    parser.add_argument("--create-splits", action="store_true",
                       help="Create train/val/test splits")
    
    args = parser.parse_args()
    
    downloader = RSUD20KDownloader(data_dir=args.data_dir)
    
    if not args.skip_download:
        success = downloader.download_partial_dataset(max_images=args.max_images)
        if not success:
            print("\n⚠ Download incomplete. Please check download instructions.")
            print("You can manually download the dataset and run with --skip-download")
            return
    
    if args.create_splits or (args.skip_download and Path(args.data_dir).exists()):
        downloader.create_splits()
        downloader.create_yolo_config()
        print("\n✅ Dataset preparation complete!")
        print(f"Dataset location: {args.data_dir}")
        print(f"YOLO config: {args.data_dir}/dataset.yaml")


if __name__ == "__main__":
    main()

