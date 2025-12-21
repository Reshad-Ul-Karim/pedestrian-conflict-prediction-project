#!/usr/bin/env python3
"""
Phase 7: Comprehensive System Evaluation
Evaluates trajectory prediction and conflict prediction performance
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, brier_score_loss
from torch.utils.data import DataLoader

from utils_config import load_config
from utils_logger import setup_logger
from utils_device import get_device


class Evaluator:
    """Evaluate conflict prediction system"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger('evaluation')
    
    def evaluate(self, model_path, test_loader, output_dir):
        """Run comprehensive evaluation"""
        self.logger.info("=" * 60)
        self.logger.info("System Evaluation")
        self.logger.info("=" * 60)
        
        device = get_device()
        
        # Load model
        from src.train_conflict_predictor import ConflictPredictor
        model = ConflictPredictor().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        self.logger.info(f"Loaded model from: {model_path}")
        
        # Collect predictions
        all_preds = []
        all_labels = []
        all_weights = []
        
        self.logger.info("Running inference...")
        with torch.no_grad():
            for batch in test_loader:
                visual = batch['visual'].to(device)
                kinematic = batch['kinematic'].to(device)
                labels = batch['labels'].numpy()
                weights = batch['weights'].numpy()
                
                pred = model(visual, kinematic).cpu().numpy()
                
                all_preds.append(pred)
                all_labels.append(labels)
                all_weights.append(weights)
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        all_weights = np.vstack(all_weights)
        
        # Compute metrics for each horizon
        results = {}
        for i, horizon in enumerate(['1s', '2s', '3s']):
            self.logger.info(f"\nHorizon: {horizon}")
            
            preds_h = all_preds[:, i]
            labels_h = all_labels[:, i]
            weights_h = all_weights[:, i]
            
            # Binary predictions (threshold=0.5)
            preds_binary = (preds_h >= 0.5).astype(int)
            
            # Classification metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_h, preds_binary, average='binary'
            )
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(labels_h, preds_h)
            except:
                roc_auc = 0.0
            
            # Brier score (calibration)
            brier = brier_score_loss(labels_h, preds_h)
            
            # Weighted metrics
            weighted_accuracy = np.average(
                (preds_binary == labels_h).astype(float),
                weights=weights_h
            )
            
            results[horizon] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'roc_auc': float(roc_auc),
                'brier_score': float(brier),
                'weighted_accuracy': float(weighted_accuracy),
                'num_positives': int(labels_h.sum()),
                'num_samples': len(labels_h)
            }
            
            self.logger.info(f"  Precision: {precision:.4f}")
            self.logger.info(f"  Recall: {recall:.4f}")
            self.logger.info(f"  F1-Score: {f1:.4f}")
            self.logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            self.logger.info(f"  Brier Score: {brier:.4f}")
            self.logger.info(f"  Weighted Accuracy: {weighted_accuracy:.4f}")
        
        # Overall metrics
        overall_f1 = np.mean([results[h]['f1'] for h in ['1s', '2s', '3s']])
        overall_auc = np.mean([results[h]['roc_auc'] for h in ['1s', '2s', '3s']])
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"Overall F1-Score: {overall_f1:.4f}")
        self.logger.info(f"Overall ROC-AUC: {overall_auc:.4f}")
        self.logger.info("=" * 60)
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"\nResults saved to: {output_dir / 'evaluation_results.json'}")
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/model.yaml')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--clips', required=True, help='Test clips directory')
    parser.add_argument('--output', default='outputs/reports/evaluation')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Load test data
    from src.train_conflict_predictor import ConflictDataset
    metadata_path = Path(args.clips) / 'metadata.json'
    dataset = ConflictDataset(args.clips, metadata_path)
    
    # Use last 15% as test set
    test_size = int(0.15 * len(dataset))
    _, test_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - test_size, test_size]
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    evaluator = Evaluator(config)
    evaluator.evaluate(args.model, test_loader, args.output)


if __name__ == "__main__":
    main()

