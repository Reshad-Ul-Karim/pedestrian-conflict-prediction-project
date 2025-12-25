#!/usr/bin/env python3
"""
Step 1.2 - Train YOLOv8 Detector on RSUD20K Dataset
Trains object detection model for multi-class detection
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml

from utils_config import load_config
from utils_logger import setup_logger
from utils_device import get_device


def train_detector(config_path: str, data_yaml: str, resume: bool = False):
    """
    Train YOLOv8 detector
    
    Args:
        config_path: Path to detector configuration
        data_yaml: Path to dataset YAML
        resume: Whether to resume training
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logger
    logger = setup_logger(
        'detector_training',
        log_file=f'{config.training.project}/{config.training.name}/train.log'
    )
    
    logger.info("=" * 60)
    logger.info("YOLOv8 Detector Training")
    logger.info("=" * 60)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Verify dataset exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    
    # Load dataset config to check paths
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    logger.info(f"Dataset: {data_config.get('path', 'N/A')}")
    logger.info(f"Classes: {data_config.get('nc', 0)}")
    logger.info(f"Class names: {data_config.get('names', [])}")
    
    # Initialize model
    model_type = config.model.type
    pretrained = config.model.pretrained
    
    logger.info(f"Model: {model_type}")
    logger.info(f"Pretrained: {pretrained}")
    
    if resume:
        # Resume from last checkpoint
        model_path = Path(config.training.project) / config.training.name / "weights" / "last.pt"
        if not model_path.exists():
            logger.warning(f"Resume checkpoint not found: {model_path}")
            logger.info("Starting training from scratch")
            model = YOLO(f"{model_type}.pt" if pretrained else f"{model_type}.yaml")
        else:
            logger.info(f"Resuming from: {model_path}")
            model = YOLO(str(model_path))
    else:
        model = YOLO(f"{model_type}.pt" if pretrained else f"{model_type}.yaml")
    
    # Training arguments
    train_args = {
        'data': str(data_path),
        'epochs': config.training.epochs,
        'batch': config.training.batch_size,
        'imgsz': config.training.imgsz,
        'device': str(device),
        'workers': config.training.workers,
        'project': config.training.project,
        'name': config.training.name,
        'exist_ok': True,
        
        # Optimizer
        'optimizer': config.training.optimizer,
        'lr0': config.training.lr0,
        'lrf': config.training.lrf,
        'momentum': config.training.momentum,
        'weight_decay': config.training.weight_decay,
        
        # Learning rate scheduler
        'warmup_epochs': config.training.warmup_epochs,
        'warmup_momentum': config.training.warmup_momentum,
        'warmup_bias_lr': config.training.warmup_bias_lr,
        
        # Augmentation
        'hsv_h': config.augmentation.hsv_h,
        'hsv_s': config.augmentation.hsv_s,
        'hsv_v': config.augmentation.hsv_v,
        'degrees': config.augmentation.degrees,
        'translate': config.augmentation.translate,
        'scale': config.augmentation.scale,
        'shear': config.augmentation.shear,
        'perspective': config.augmentation.perspective,
        'flipud': config.augmentation.flipud,
        'fliplr': config.augmentation.fliplr,
        'mosaic': config.augmentation.mosaic,
        'mixup': config.augmentation.mixup,
        'copy_paste': config.augmentation.copy_paste,
        
        # Validation
        'val': True,
        'save': config.validation.save_best,
        'save_period': config.validation.save_period,
        'patience': config.validation.patience,
        
        # Logging
        'verbose': config.logging.get('verbose', True),
        'plots': config.logging.get('plots', True),
    }
    
    logger.info("\nTraining Configuration:")
    for key, value in train_args.items():
        logger.info(f"  {key}: {value}")
    
    # Train model
    logger.info("\nStarting training...")
    results = model.train(**train_args)
    
    logger.info("\nTraining completed!")
    logger.info(f"Best model saved to: {model.trainer.best}")
    logger.info(f"Last model saved to: {model.trainer.last}")
    
    # Validation results
    if hasattr(results, 'results_dict'):
        logger.info("\nValidation Results:")
        for metric, value in results.results_dict.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Export best model info
    output_info = {
        'model_type': model_type,
        'best_model_path': str(model.trainer.best),
        'last_model_path': str(model.trainer.last),
        'dataset': str(data_path),
        'device': str(device),
    }
    
    output_file = Path(config.training.project) / config.training.name / "train_info.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(output_info, f, default_flow_style=False)
    
    logger.info(f"\nTraining info saved to: {output_file}")
    logger.info("=" * 60)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/detector.yaml',
        help='Path to detector configuration'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    
    args = parser.parse_args()
    
    try:
        train_detector(args.config, args.data, args.resume)
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()

