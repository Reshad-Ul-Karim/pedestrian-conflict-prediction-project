#!/usr/bin/env python3
"""
Phase 6: Train Conflict Prediction Fusion Model
Vision + Kinematics fusion for multi-horizon conflict prediction
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from utils_config import load_config
from utils_logger import setup_logger
from utils_device import get_device


class ConflictDataset(Dataset):
    """Dataset for conflict prediction"""
    
    def __init__(self, clips_dir, metadata_path):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.clips_dir = Path(clips_dir)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load visual clip
        visual = np.load(self.clips_dir / item['visual_path'])
        visual = torch.from_numpy(visual).float().permute(0, 3, 1, 2) / 255.0
        
        # Load kinematic sequence
        kin = np.load(self.clips_dir / item['kinematic_path'])
        kin = torch.from_numpy(kin).float()
        
        # Labels
        labels = item['labels']
        y_1s = labels['y_1s']
        y_2s = labels['y_2s']
        y_3s = labels['y_3s']
        w_1s = labels['w_1s']
        w_2s = labels['w_2s']
        w_3s = labels['w_3s']
        
        return {
            'visual': visual,
            'kinematic': kin,
            'labels': torch.tensor([y_1s, y_2s, y_3s], dtype=torch.float),
            'weights': torch.tensor([w_1s, w_2s, w_3s], dtype=torch.float)
        }


class VisionEncoder(nn.Module):
    """3D CNN for visual encoding"""
    
    def __init__(self, output_dim=512):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(256, output_dim)
    
    def forward(self, x):
        # x: [batch, time, channels, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [batch, channels, time, H, W]
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class KinematicsEncoder(nn.Module):
    """BiLSTM for kinematic encoding"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)  # Concatenate bidirectional
        return self.fc(h_n)


class ConflictPredictor(nn.Module):
    """Fusion model for conflict prediction"""
    
    def __init__(self, vision_dim=512, kin_dim=256, hidden_dim=256):
        super().__init__()
        
        self.vision_encoder = VisionEncoder(vision_dim)
        self.kin_encoder = KinematicsEncoder(output_dim=kin_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + kin_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Prediction heads
        self.head_1s = nn.Linear(128, 1)
        self.head_2s = nn.Linear(128, 1)
        self.head_3s = nn.Linear(128, 1)
    
    def forward(self, visual, kinematic):
        v_feat = self.vision_encoder(visual)
        k_feat = self.kin_encoder(kinematic)
        
        fused = torch.cat([v_feat, k_feat], dim=1)
        h = self.fusion(fused)
        
        p_1s = torch.sigmoid(self.head_1s(h))
        p_2s = torch.sigmoid(self.head_2s(h))
        p_3s = torch.sigmoid(self.head_3s(h))
        
        return torch.cat([p_1s, p_2s, p_3s], dim=1)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def train_model(config_path, clips_dir, output_dir):
    """Train conflict prediction model"""
    config = load_config(config_path)
    logger = setup_logger('conflict_training', log_file=f'{output_dir}/train.log')
    
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    metadata_path = Path(clips_dir) / 'metadata.json'
    dataset = ConflictDataset(clips_dir, metadata_path)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size,
                             shuffle=True, num_workers=config.training.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = ConflictPredictor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                           weight_decay=config.training.weight_decay)
    
    if config.training.loss.type == 'focal':
        criterion = FocalLoss(alpha=config.training.loss.focal_alpha,
                            gamma=config.training.loss.focal_gamma)
    else:
        criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config.training.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            visual = batch['visual'].to(device)
            kinematic = batch['kinematic'].to(device)
            labels = batch['labels'].to(device)
            weights = batch['weights'].to(device)
            
            optimizer.zero_grad()
            pred = model(visual, kinematic)
            
            # Weighted loss
            if config.training.loss.use_confidence_weights:
                loss = (criterion(pred, labels) * weights).mean()
            else:
                loss = criterion(pred, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                visual = batch['visual'].to(device)
                kinematic = batch['kinematic'].to(device)
                labels = batch['labels'].to(device)
                
                pred = model(visual, kinematic)
                loss = criterion(pred, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
            logger.info(f"Saved best model with val loss: {val_loss:.4f}")
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/model.yaml')
    parser.add_argument('--clips', required=True, help='Training clips directory')
    parser.add_argument('--output', default='outputs/models/conflict_predictor')
    args = parser.parse_args()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    train_model(args.config, args.clips, args.output)


if __name__ == "__main__":
    main()

