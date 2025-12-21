#!/usr/bin/env python3
"""
Phase 8: Model Distillation and Export
Distill to lightweight model and export to ONNX/CoreML
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn

from utils_config import load_config
from utils_logger import setup_logger
from utils_device import get_device, convert_to_coreml


class LightweightPredictor(nn.Module):
    """Lightweight student model"""
    
    def __init__(self):
        super().__init__()
        
        # Simplified vision encoder
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128)
        )
        
        # Simplified kinematics encoder
        self.kinematics = nn.GRU(6, 32, batch_first=True)
        
        # Fusion and heads
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, visual, kinematic):
        # Visual: [batch, frames, C, H, W] -> use middle frame
        v_feat = self.vision(visual[:, visual.size(1)//2])
        
        # Kinematic
        _, k_feat = self.kinematics(kinematic)
        k_feat = k_feat.squeeze(0)
        
        # Fusion
        fused = torch.cat([v_feat, k_feat], dim=1)
        out = torch.sigmoid(self.fusion(fused))
        
        return out


def distill_model(teacher_path, clips_dir, output_dir):
    """Distill teacher model to student"""
    logger = setup_logger('distillation', log_file=f'{output_dir}/distill.log')
    logger.info("Model Distillation")
    
    device = get_device()
    
    # Load teacher
    from src.train_conflict_predictor import ConflictPredictor, ConflictDataset
    teacher = ConflictPredictor().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    
    # Create student
    student = LightweightPredictor().to(device)
    
    # Distillation training (simplified)
    from torch.utils.data import DataLoader
    metadata_path = Path(clips_dir) / 'metadata.json'
    dataset = ConflictDataset(clips_dir, metadata_path)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    temperature = 5.0
    alpha = 0.7
    
    logger.info("Training student model...")
    for epoch in range(10):
        total_loss = 0.0
        for batch in train_loader:
            visual = batch['visual'].to(device)
            kinematic = batch['kinematic'].to(device)
            labels = batch['labels'].to(device)
            
            # Teacher predictions
            with torch.no_grad():
                teacher_pred = teacher(visual, kinematic)
            
            # Student predictions
            student_pred = student(visual, kinematic)
            
            # Distillation loss
            loss_hard = nn.functional.binary_cross_entropy(student_pred, labels)
            loss_soft = nn.functional.kl_div(
                torch.log(student_pred / temperature + 1e-10),
                teacher_pred / temperature,
                reduction='batchmean'
            )
            
            loss = alpha * loss_hard + (1 - alpha) * temperature**2 * loss_soft
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")
    
    # Save student
    torch.save(student.state_dict(), f'{output_dir}/student_model.pt')
    logger.info(f"Saved student model")
    
    # Export to ONNX
    example_visual = torch.randn(1, 20, 3, 128, 128).to(device)
    example_kin = torch.randn(1, 20, 6).to(device)
    
    onnx_path = f'{output_dir}/conflict_predictor.onnx'
    torch.onnx.export(
        student,
        (example_visual, example_kin),
        onnx_path,
        input_names=['visual', 'kinematic'],
        output_names=['predictions'],
        opset_version=17
    )
    logger.info(f"Exported to ONNX: {onnx_path}")
    
    # Export to CoreML
    try:
        mlmodel = convert_to_coreml(
            student,
            example_visual,
            f'{output_dir}/conflict_predictor.mlmodel'
        )
        logger.info(f"Exported to CoreML")
    except Exception as e:
        logger.warning(f"CoreML export failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', required=True, help='Teacher model path')
    parser.add_argument('--clips', required=True)
    parser.add_argument('--output', default='outputs/models/distilled')
    args = parser.parse_args()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    distill_model(args.teacher, args.clips, args.output)


if __name__ == "__main__":
    main()

