#!/usr/bin/env python3
"""
Phase 3: Trajectory Prediction Model
Transformer-based multi-horizon trajectory prediction
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


class TrajectoryDataset(Dataset):
    """Dataset for trajectory prediction"""
    
    def __init__(self, trajectories_path, past_frames=20, future_horizons=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
        self.past_frames = past_frames
        self.future_horizons = future_horizons
        self.samples = []
        
        # Load and process trajectories
        with open(trajectories_path, 'r') as f:
            for line in f:
                traj = json.loads(line)
                self._process_trajectory(traj)
    
    def _process_trajectory(self, traj):
        """Extract sliding windows from trajectory"""
        trajectory_data = traj['trajectory']
        if len(trajectory_data) < self.past_frames + len(self.future_horizons):
            return
        
        for i in range(len(trajectory_data) - self.past_frames - len(self.future_horizons)):
            past_window = trajectory_data[i:i + self.past_frames]
            
            # Extract features
            features = []
            for point in past_window:
                feat = [
                    point['u'] / 1280.0,  # Normalize
                    point['v'] / 720.0,
                    point['kinematics']['v_u'] if point.get('kinematics') else 0.0,
                    point['kinematics']['v_v'] if point.get('kinematics') else 0.0,
                    point['kinematics']['heading'] if point.get('kinematics') else 0.0,
                ]
                features.append(feat)
            
            # Extract future positions
            future_positions = []
            base_idx = i + self.past_frames
            for horizon_idx in range(len(self.future_horizons)):
                if base_idx + horizon_idx < len(trajectory_data):
                    fut_point = trajectory_data[base_idx + horizon_idx]
                    future_positions.append([fut_point['u'] / 1280.0, fut_point['v'] / 720.0])
            
            if len(future_positions) == len(self.future_horizons):
                self.samples.append({
                    'past': np.array(features, dtype=np.float32),
                    'future': np.array(future_positions, dtype=np.float32)
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.from_numpy(sample['past']), torch.from_numpy(sample['future'])


class TrajectoryTransformer(nn.Module):
    """Transformer-based trajectory prediction model"""
    
    def __init__(self, input_dim=5, d_model=128, nhead=4, num_layers=3, num_horizons=6):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_horizons * 2)  # x, y for each horizon
        )
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        pred = self.decoder(x)
        return pred.view(-1, 6, 2)  # [batch, num_horizons, 2]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def train_model(config_path, trajectories_path, output_dir):
    """Train trajectory prediction model"""
    config = load_config(config_path)
    logger = setup_logger('trajectory_training', log_file=f'{output_dir}/train.log')
    
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = TrajectoryDataset(trajectories_path, past_frames=config.prediction.past_frames)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.prediction.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.prediction.batch_size)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = TrajectoryTransformer(
        d_model=config.prediction.transformer.d_model,
        nhead=config.prediction.transformer.nhead,
        num_layers=config.prediction.transformer.num_encoder_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.prediction.learning_rate,
                          weight_decay=config.prediction.weight_decay)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.prediction.epochs):
        model.train()
        train_loss = 0.0
        
        for past, future in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            past, future = past.to(device), future.to(device)
            
            optimizer.zero_grad()
            pred = model(past)
            loss = criterion(pred, future)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                pred = model(past)
                loss = criterion(pred, future)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
    
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/trajectory.yaml')
    parser.add_argument('--trajectories', required=True)
    parser.add_argument('--output', default='outputs/models/trajectory_predictor')
    args = parser.parse_args()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    train_model(args.config, args.trajectories, args.output)


if __name__ == "__main__":
    main()

