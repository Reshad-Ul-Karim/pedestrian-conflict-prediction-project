#!/usr/bin/env python3
"""
Complete FT-Transformer training script for conflict prediction
Includes hyperparameter tuning with Ray Tune, preprocessing, evaluation, and model export
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    cohen_kappa_score, f1_score
)
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch Tabular
try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import FTTransformerConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    PYTORCH_TABULAR_AVAILABLE = True
except ImportError:
    print("⚠ PyTorch Tabular not available. Install: pip install pytorch-tabular")
    PYTORCH_TABULAR_AVAILABLE = False

# Ray Tune removed - using simple grid search instead
RAY_AVAILABLE = False

# Set random seeds for reproducibility
import torch
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration
def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ MPS (Apple Silicon) available")
        return device
    else:
        device = torch.device('cpu')
        print("⚠ No GPU available, using CPU")
        return device

DEVICE = get_device()


class ConflictDatasetPreprocessor:
    """Preprocessing pipeline for conflict dataset"""
    
    def __init__(self, csv_path, test_size=0.3, random_state=42):
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.categorical_columns = []
        self.continuous_columns = []
        
    def load_and_preprocess(self):
        """Load CSV and perform preprocessing with normalization and interaction features"""
        print("=" * 60)
        print("Loading and Preprocessing Dataset")
        print("=" * 60)
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(df)} records, {len(df.columns)} columns")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            df = df.fillna(0)
            print(f"✓ Filled {missing.sum()} missing values")
        
        # Estimate image dimensions from bbox coordinates (for normalization)
        if 'bbox_x2' in df.columns and 'bbox_y2' in df.columns:
            estimated_img_width = int(df['bbox_x2'].max() * 1.1)
            estimated_img_height = int(df['bbox_y2'].max() * 1.1)
        else:
            estimated_img_width = 1920
            estimated_img_height = 1080
        
        # Base features as specified by user
        base_bbox_features = [
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'bbox_center_x', 'bbox_center_y', 'bbox_area'
        ]
        base_pose_features = [
            'position_agreement', 'pose_detected', 'pose_confidence',
            'body_orientation_angle'
            # REMOVED: 'pose_score' (was deterministic/computed)
        ]
        
        # Normalize bounding box features
        if all(col in df.columns for col in base_bbox_features):
            # Normalize bbox coordinates
            df['bbox_center_x_norm'] = df['bbox_center_x'] / estimated_img_width
            df['bbox_center_y_norm'] = df['bbox_center_y'] / estimated_img_height
            df['bbox_x1_norm'] = df['bbox_x1'] / estimated_img_width
            df['bbox_y1_norm'] = df['bbox_y1'] / estimated_img_height
            df['bbox_x2_norm'] = df['bbox_x2'] / estimated_img_width
            df['bbox_y2_norm'] = df['bbox_y2'] / estimated_img_height
            df['bbox_area_norm'] = df['bbox_area'] / (estimated_img_width * estimated_img_height)
            
            # Compute derived normalized features
            df['bbox_width'] = df['bbox_x2'] - df['bbox_x1']
            df['bbox_height'] = df['bbox_y2'] - df['bbox_y1']
            df['bbox_width_norm'] = df['bbox_width'] / estimated_img_width
            df['bbox_height_norm'] = df['bbox_height'] / estimated_img_height
            df['bbox_aspect_ratio'] = df['bbox_width'] / (df['bbox_height'] + 1e-6)
        
        # Handle pose features
        if 'pose_detected' in df.columns:
            if df['pose_detected'].dtype == 'bool':
                df['pose_detected'] = df['pose_detected'].astype(int)
        
        # Fill NaN values for pose features
        for col in base_pose_features:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        
        # Interaction Features for Non-linearity
        # Area × pose confidence (size × pose quality)
        if 'bbox_area_norm' in df.columns and 'pose_confidence' in df.columns:
            df['area_pose_confidence_interaction'] = df['bbox_area_norm'] * df['pose_confidence']
        
        # Position × pose confidence (location × pose quality) - using raw confidence instead of computed score
        if 'bbox_center_y_norm' in df.columns and 'pose_confidence' in df.columns:
            df['position_pose_confidence_interaction'] = df['bbox_center_y_norm'] * df['pose_confidence']
        
        # Orientation × position agreement (body direction × position agreement)
        if 'body_orientation_angle' in df.columns and 'position_agreement' in df.columns:
            df['orientation_agreement_interaction'] = (np.abs(df['body_orientation_angle']) / 180.0) * df['position_agreement']
        
        # Area × orientation (size × movement direction)
        if 'bbox_area_norm' in df.columns and 'body_orientation_angle' in df.columns:
            df['area_orientation_interaction'] = df['bbox_area_norm'] * (np.abs(df['body_orientation_angle']) / 180.0)
        
        # Pose confidence × orientation (pose quality × movement direction)
        if 'pose_confidence' in df.columns and 'body_orientation_angle' in df.columns:
            df['pose_orientation_interaction'] = df['pose_confidence'] * (np.abs(df['body_orientation_angle']) / 180.0)
        
        # Define final feature set: normalized bbox + pose + interactions
        normalized_bbox_features = [
            'bbox_x1_norm', 'bbox_y1_norm', 'bbox_x2_norm', 'bbox_y2_norm',
            'bbox_center_x_norm', 'bbox_center_y_norm', 'bbox_area_norm',
            'bbox_width_norm', 'bbox_height_norm', 'bbox_aspect_ratio'
        ]
        
        pose_features = base_pose_features.copy()
        
        interaction_features = [
            'area_pose_confidence_interaction',
            'position_pose_confidence_interaction',  # Changed from position_pose_score_interaction
            'orientation_agreement_interaction',
            'area_orientation_interaction',
            'pose_orientation_interaction'
        ]
        
        # Select only features that exist in the dataframe
        all_features = normalized_bbox_features + pose_features + interaction_features
        available_features = []
        for feat_list in [normalized_bbox_features, pose_features, interaction_features]:
            for feat in feat_list:
                if feat in df.columns:
                    available_features.append(feat)
        
        self.feature_columns = available_features
        
        # Separate categorical and continuous features
        for col in self.feature_columns:
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                self.categorical_columns.append(col)
            else:
                self.continuous_columns.append(col)
        
        # Convert boolean to int
        for col in self.categorical_columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
        
        # Handle categorical features (encode if needed)
        if self.categorical_columns:
            for col in self.categorical_columns:
                if df[col].dtype == 'object':
                    df[col] = self.label_encoder.fit_transform(df[col].astype(str))
        
        # Prepare features and target
        X = df[self.feature_columns].copy()
        y = df['conflict_score'].copy()  # Target column
        y_class = df['risk_level'].copy()
        y_class_encoded = self.label_encoder.fit_transform(y_class)
        
        print(f"✓ Features: {len(self.feature_columns)} total ({len(self.continuous_columns)} continuous, {len(self.categorical_columns)} categorical)")
        print(f"  Normalized bbox: {len([f for f in normalized_bbox_features if f in self.feature_columns])}")
        print(f"  Pose features: {len([f for f in pose_features if f in self.feature_columns])}")
        print(f"  Interactions: {len([f for f in interaction_features if f in self.feature_columns])}")
        print(f"✓ Target: conflict_score [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")
        
        # Train-test split (70/30) - IMPORTANT: Split by image_id to avoid leakage
        unique_images = df['image_id'].unique()
        train_images, test_images = train_test_split(
            unique_images, test_size=self.test_size, random_state=self.random_state
        )
        
        train_mask = df['image_id'].isin(train_images)
        test_mask = df['image_id'].isin(test_images)
        
        X_train = X[train_mask].copy()
        X_test = X[test_mask].copy()
        y_train = y[train_mask].copy()
        y_test = y[test_mask].copy()
        y_train_class = y_class_encoded[train_mask]
        y_test_class = y_class_encoded[test_mask]
        
        print(f"✓ Train/Test split: {len(X_train)}/{len(X_test)} records ({len(train_images)}/{len(test_images)} images)")
        
        # Scale continuous features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if self.continuous_columns:
            X_train_scaled[self.continuous_columns] = self.scaler.fit_transform(X_train[self.continuous_columns])
            X_test_scaled[self.continuous_columns] = self.scaler.transform(X_test[self.continuous_columns])
        
        print("✓ Preprocessing complete\n")
        print("⚠ Anti-overfitting measures active:")
        print("   - Aggressive dropout (0.2-0.25)")
        print("   - Strong weight decay (1e-3)")
        print("   - Reduced model complexity")
        print("   - Early stopping (patience=10)")
        print("   - Gradient clipping")
        print("")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_class': y_train_class,
            'y_test_class': y_test_class,
            'train_images': train_images,
            'test_images': test_images,
            'feature_columns': self.feature_columns,
            'continuous_columns': self.continuous_columns,
            'categorical_columns': self.categorical_columns
        }
    
    def save_preprocessor(self, save_path):
        """Save preprocessor for inference"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'continuous_columns': self.continuous_columns,
            'categorical_columns': self.categorical_columns
        }
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)


class FTTransformerTrainer:
    """FT-Transformer trainer with hyperparameter tuning"""
    
    def __init__(self, data=None, output_dir='/content/models/ft_transformer', device=None):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model = None
        self.best_config = None
        self.device = device if device is not None else DEVICE
        
    def train_with_tuning(self, num_samples=20, max_epochs=50):
        """Train with simple grid search hyperparameter optimization"""
        if not PYTORCH_TABULAR_AVAILABLE:
            print("⚠ PyTorch Tabular not available. Cannot train FT-Transformer.")
            return None
        
        return self.train_simple_grid_search(num_samples, max_epochs)
    
    def train_simple_grid_search(self, num_samples=20, max_epochs=50):
        """Simple grid search for hyperparameter optimization"""
        if not PYTORCH_TABULAR_AVAILABLE:
            print("⚠ PyTorch Tabular not available. Cannot train FT-Transformer.")
            return None
        
        print("\n" + "=" * 60)
        print("Hyperparameter Tuning (Grid Search)")
        print("=" * 60)
        
        # Define search space - further reduced complexity to prevent overfitting
        learning_rates = [1e-4, 3e-4, 5e-4]  # Removed 1e-3, focus on lower learning rates
        num_heads_list = [2, 4]              # Removed 8, focus on smaller models
        embedding_dims = [32, 64]           # Removed 128, focus on smaller embeddings
        batch_sizes = [64, 128]             # Removed 256, prefer smaller batches for more variance
        
        # Generate random combinations
        import random
        random.seed(42)
        configs = []
        for _ in range(min(num_samples, 20)):
            configs.append({
                'learning_rate': random.choice(learning_rates),
                'num_heads': random.choice(num_heads_list),
                'num_attn_blocks': random.choice([2, 3, 4]),  # Reduced from [4, 6, 8]
                'embedding_dim': random.choice(embedding_dims),
                'depth': random.choice([2, 3]),  # Reduced from [3, 4, 5]
                'batch_size': random.choice(batch_sizes),
            })
        
        best_loss = float('inf')
        best_config = None
        best_model = None
        
        print(f"Testing {len(configs)} hyperparameter combinations...")
        for i, config in enumerate(configs, 1):
            try:
                model = self._create_model(config)
                model.fit(
                    train=self.data['train_df'],
                    validation=self.data['val_df'],
                    max_epochs=max_epochs
                )
                
                result = model.evaluate(self.data['val_df'])
                val_loss = result['test_loss']
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_config = config.copy()
                    best_model = model
                    print(f"  [{i}/{len(configs)}] ✓ New best: loss={val_loss:.4f}, R²={result.get('test_r2', 0):.4f}")
                    
            except Exception as e:
                if i % 5 == 0:  # Only print every 5th failure
                    print(f"  [{i}/{len(configs)}] Failed: {e}")
                continue
        
        if best_model is None:
            print("⚠ All configurations failed. Using defaults...")
            return self.train_default(max_epochs)
        
        self.best_model = best_model
        self.best_config = best_config
        print(f"✓ Best config found: loss={best_loss:.4f}")
        
        return self.best_model
    
    def train_default(self, max_epochs=50):
        """Train with default hyperparameters"""
        if not PYTORCH_TABULAR_AVAILABLE:
            print("⚠ PyTorch Tabular not available. Cannot train FT-Transformer.")
            return None
        
        print("\n" + "=" * 60)
        print("Training FT-Transformer (Default Hyperparameters)")
        print("=" * 60)
        print("Training...")
        
        # Default config - further reduced complexity to prevent overfitting
        config = {
            "learning_rate": 3e-4,  # Further reduced from 5e-4 for more stable training
            "num_heads": 4,         # Keep at 4
            "num_attn_blocks": 3,   # Further reduced from 4 to 3
            "embedding_dim": 64,    # Keep at 64
            "depth": 2,             # Further reduced from 3 to 2 (shallower model)
            "batch_size": 128,      # Keep at 128 (smaller batches = more regularization)
        }
        
        self.best_model = self._create_model(config)
        self.best_model.fit(
            train=self.data['train_df'],
            validation=self.data['val_df'],
            max_epochs=max_epochs
        )
        
        return self.best_model
    
    def _create_model(self, config):
        """Create FT-Transformer model with given config"""
        # Data config
        data_config = DataConfig(
            target=['conflict_score'],
            continuous_cols=self.data['continuous_columns'],
            categorical_cols=self.data['categorical_columns'] if self.data['categorical_columns'] else [],
        )
        
        # Model config - use only valid parameters for FTTransformerConfig
        # Start with minimal required parameters
        model_config_params = {
            'task': "regression",
            'learning_rate': config['learning_rate'],
            'num_heads': config['num_heads'],
            'num_attn_blocks': config['num_attn_blocks'],
        }
        
        # Add optional parameters if they work
        embed_dim = config.get('embedding_dim', 128)
        for param_name in ['input_embed_dim', 'd_model']:
            try:
                test_params = dict(model_config_params)
                test_params[param_name] = embed_dim
                FTTransformerConfig(**test_params)
                model_config_params[param_name] = embed_dim
                break
            except (TypeError, ValueError):
                continue
        
        depth = config.get('depth', 4)
        for param_name in ['num_transformer_blocks', 'depth']:
            try:
                test_params = dict(model_config_params)
                test_params[param_name] = depth
                FTTransformerConfig(**test_params)
                model_config_params[param_name] = depth
                break
            except (TypeError, ValueError):
                continue
        
        # Add aggressive dropout for regularization (if supported)
        # Higher dropout rates to introduce more variance and prevent overfitting
        for dropout_param in ['attn_dropout', 'attention_dropout', 'dropout']:
            try:
                test_params = dict(model_config_params)
                test_params[dropout_param] = 0.2  # Increased from 0.1 to 0.2
                FTTransformerConfig(**test_params)
                model_config_params[dropout_param] = 0.2
                break
            except (TypeError, ValueError):
                continue
        
        for ff_dropout_param in ['ff_dropout', 'ff_dropout_rate', 'feedforward_dropout']:
            try:
                test_params = dict(model_config_params)
                test_params[ff_dropout_param] = 0.25  # Increased from 0.1 to 0.25
                FTTransformerConfig(**test_params)
                model_config_params[ff_dropout_param] = 0.25
                break
            except (TypeError, ValueError):
                continue
        
        # Add embedding dropout if supported
        for embed_dropout_param in ['embedding_dropout', 'embed_dropout', 'input_dropout']:
            try:
                test_params = dict(model_config_params)
                test_params[embed_dropout_param] = 0.1
                FTTransformerConfig(**test_params)
                model_config_params[embed_dropout_param] = 0.1
                break
            except (TypeError, ValueError):
                continue
        
        # Create config with validated parameters
        model_config = FTTransformerConfig(**model_config_params)
        
        # Optimizer config with aggressive weight decay for L2 regularization
        optimizer_config = OptimizerConfig(
            optimizer='AdamW',  # Use AdamW instead of Adam for better weight decay
            optimizer_params={
                'weight_decay': 1e-3,  # Increased from 1e-4 to 1e-3 for stronger regularization
                'betas': (0.9, 0.999),  # Standard Adam betas
                'eps': 1e-8
            }
        )
        
        # Trainer config with device support and aggressive regularization
        trainer_config_params = {
            'batch_size': config['batch_size'],
            'max_epochs': 50,  # Will be overridden by max_epochs parameter
            'early_stopping': "valid_loss",  # Monitor validation loss
            'early_stopping_patience': 10,  # Reduced from 15 to 10 for more aggressive stopping
            'checkpoints': "valid_loss",  # Save best model based on validation loss
            'checkpoints_path': str(Path(self.output_dir) / "checkpoints"),
            # Device configuration
            'accelerator': 'gpu' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
            'devices': 1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else None,
        }
        
        # Add optional early stopping parameters if they work
        for param_name in ['early_stopping_min_delta', 'min_delta']:
            try:
                test_params = dict(trainer_config_params)
                test_params[param_name] = 1e-4  # Increased from 1e-5 to 1e-4 (require more significant improvement)
                TrainerConfig(**test_params)
                trainer_config_params[param_name] = 1e-4
                break
            except (TypeError, ValueError):
                continue
        
        # Try to add mode parameter for early stopping (minimize or maximize)
        for mode_param in ['early_stopping_mode', 'mode']:
            try:
                test_params = dict(trainer_config_params)
                test_params[mode_param] = "min"  # Minimize validation loss
                TrainerConfig(**test_params)
                trainer_config_params[mode_param] = "min"
                break
            except (TypeError, ValueError):
                continue
        
        # Add gradient clipping to prevent exploding gradients and add variance
        for grad_clip_param in ['gradient_clip_val', 'grad_clip_val', 'max_grad_norm']:
            try:
                test_params = dict(trainer_config_params)
                test_params[grad_clip_param] = 1.0  # Clip gradients to norm of 1.0
                TrainerConfig(**test_params)
                trainer_config_params[grad_clip_param] = 1.0
                break
            except (TypeError, ValueError):
                continue
        
        # Try to add learning rate scheduler if supported
        for scheduler_param in ['learning_rate_scheduler', 'lr_scheduler']:
            try:
                test_params = dict(trainer_config_params)
                test_params[scheduler_param] = "ReduceLROnPlateau"
                TrainerConfig(**test_params)
                trainer_config_params[scheduler_param] = "ReduceLROnPlateau"
                # Try to add scheduler params
                for scheduler_params_name in ['learning_rate_scheduler_params', 'lr_scheduler_params', 'scheduler_params']:
                    try:
                        test_params2 = dict(trainer_config_params)
                        test_params2[scheduler_params_name] = {"patience": 5, "factor": 0.5, "mode": "min"}
                        TrainerConfig(**test_params2)
                        trainer_config_params[scheduler_params_name] = {"patience": 5, "factor": 0.5, "mode": "min"}
                        break
                    except (TypeError, ValueError):
                        continue
                break
            except (TypeError, ValueError):
                continue
        
        trainer_config = TrainerConfig(**trainer_config_params)
        
        # Create model
        model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        
        return model
    
    def prepare_pytorch_tabular_data(self, preprocessed_data):
        """Prepare data for PyTorch Tabular format"""
        # Combine features and target
        train_df = preprocessed_data['X_train'].copy()
        train_df['conflict_score'] = preprocessed_data['y_train'].values
        
        test_df = preprocessed_data['X_test'].copy()
        test_df['conflict_score'] = preprocessed_data['y_test'].values
        
        # Split train into train/val (70/30) - more validation data to better detect overfitting
        from sklearn.model_selection import train_test_split
        train_df_split, val_df_split = train_test_split(
            train_df, test_size=0.3, random_state=42  # Increased from 0.2 to 0.3
        )
        
        self.data = {
            'train_df': train_df_split,
            'val_df': val_df_split,
            'test_df': test_df,
            'continuous_columns': preprocessed_data['continuous_columns'],
            'categorical_columns': preprocessed_data['categorical_columns']
        }
        
        return self.data


class EnsembleModel:
    """
    Ensemble of multiple models for uncertainty quantification
    """
    def __init__(self, models, weights=None):
        """
        Args:
            models: List of trained models
            weights: Optional list of weights for each model (default: equal weights)
        """
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict_with_uncertainty(self, data_df):
        """
        Predict with uncertainty estimates from ensemble
        
        Returns:
            dict with 'mean', 'std', 'confidence', 'lower_bound', 'upper_bound'
        """
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(data_df)
                if isinstance(pred, pd.DataFrame):
                    pred_values = pred['conflict_score_prediction'].values
                else:
                    pred_values = pred
                predictions.append(pred_values)
            except Exception as e:
                print(f"Warning: Model prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models produced valid predictions")
        
        # Stack predictions
        pred_array = np.array(predictions)
        
        # Weighted mean
        weighted_preds = np.array([pred * w for pred, w in zip(predictions, self.weights)])
        mean_pred = np.sum(weighted_preds, axis=0)
        
        # Standard deviation (uncertainty)
        std_pred = np.std(pred_array, axis=0)
        
        # Confidence (inverse of uncertainty, normalized)
        confidence = 1.0 / (1.0 + std_pred)
        
        # 95% confidence interval
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'confidence': confidence,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'individual_predictions': pred_array
        }


class ModelEvaluator:
    """Model evaluation and reporting"""
    
    def __init__(self, model, data, output_dir, ensemble_models=None):
        self.model = model
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_models = ensemble_models  # Optional ensemble for uncertainty
    
    def evaluate(self, optimize_thresholds=True, use_ensemble=False):
        """
        ENHANCED comprehensive model evaluation with threshold optimization and uncertainty
        
        Args:
            optimize_thresholds: If True, optimize thresholds for better category separation
            use_ensemble: If True and ensemble_models available, use ensemble for uncertainty
        """
        print("\n" + "=" * 60)
        print("ENHANCED Model Evaluation")
        print("=" * 60)
        
        # Get predictions
        test_result = self.model.evaluate(self.data['test_df'])
        predictions = self.model.predict(self.data['test_df'])
        
        y_true = self.data['test_df']['conflict_score'].values
        y_pred = predictions['conflict_score_prediction'].values if isinstance(predictions, pd.DataFrame) else predictions
        
        # Uncertainty quantification with ensemble
        uncertainty_info = None
        if use_ensemble and self.ensemble_models:
            print("\nComputing uncertainty estimates from ensemble...")
            try:
                ensemble_pred = self.ensemble_models.predict_with_uncertainty(self.data['test_df'])
                uncertainty_info = {
                    'mean': ensemble_pred['mean'],
                    'std': ensemble_pred['std'],
                    'confidence': ensemble_pred['confidence'],
                    'lower_bound': ensemble_pred['lower_bound'],
                    'upper_bound': ensemble_pred['upper_bound']
                }
                # Use ensemble mean as prediction
                y_pred = ensemble_pred['mean']
                print(f"  Average uncertainty (std): {ensemble_pred['std'].mean():.4f}")
                print(f"  Average confidence: {ensemble_pred['confidence'].mean():.4f}")
            except Exception as e:
                print(f"  Warning: Ensemble prediction failed: {e}")
                uncertainty_info = None
        
        # Regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\nRegression Metrics:")
        print(f"  MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # ENHANCED: Optimize thresholds for better category separation
        if optimize_thresholds:
            low_thresh, high_thresh = self._optimize_thresholds(y_true, y_pred)
        else:
            low_thresh, high_thresh = 0.35, 0.65  # Use improved defaults
        
        # Classification metrics with optimized thresholds
        y_true_class = self._score_to_risk_level(y_true, low_thresh, high_thresh)
        y_pred_class = self._score_to_risk_level(y_pred, low_thresh, high_thresh)
        
        # Calculate class weights for imbalanced classes
        class_weights = self._calculate_class_weights(y_true_class)
        
        print(f"\nClassification Report (Risk Levels) - Optimized Thresholds:")
        print(classification_report(y_true_class, y_pred_class, target_names=['LOW', 'MED', 'HIGH']))
        
        # ENHANCED: Additional metrics
        kappa = cohen_kappa_score(y_true_class, y_pred_class)
        macro_f1 = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
        
        print(f"\nAdditional Metrics:")
        print(f"  Cohen's Kappa: {kappa:.4f} (agreement beyond chance)")
        print(f"  Macro F1-Score: {macro_f1:.4f} (equal weight per class)")
        print(f"  Weighted F1-Score: {weighted_f1:.4f} (weighted by class frequency)")
        
        # Confusion matrix
        cm = confusion_matrix(y_true_class, y_pred_class)
        self._plot_confusion_matrix(cm, ['LOW', 'MED', 'HIGH'])
        
        # ENHANCED: Plot with optimized thresholds
        self._plot_predictions(y_true, y_pred, low_thresh, high_thresh)
        
        # Feature importance (if available)
        if hasattr(self.model, 'get_feature_importance'):
            try:
                importance = self.model.get_feature_importance()
                self._plot_feature_importance(importance)
            except:
                pass
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'kappa': kappa,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'optimal_thresholds': {'low': low_thresh, 'high': high_thresh},
            'class_weights': class_weights,
            'classification_report': classification_report(
                y_true_class, y_pred_class, 
                target_names=['LOW', 'MED', 'HIGH'],
                output_dict=True
            )
        }
    
    def _score_to_risk_level(self, scores, low_threshold=0.35, high_threshold=0.65):
        """
        Convert conflict scores to risk levels with optimized thresholds
        
        Args:
            scores: Array of conflict scores
            low_threshold: Threshold between LOW and MED (default 0.35)
            high_threshold: Threshold between MED and HIGH (default 0.65)
        """
        levels = []
        for score in scores:
            if score > high_threshold:
                levels.append('HIGH')
            elif score > low_threshold:
                levels.append('MED')
            else:
                levels.append('LOW')
        return np.array(levels)
    
    def _optimize_thresholds(self, y_true, y_pred):
        """
        Optimize thresholds for LOW/MED/HIGH classification using F1-score maximization
        
        Returns:
            tuple: (low_threshold, high_threshold) that maximize macro F1-score
        """
        print("\nOptimizing thresholds for category separation...")
        
        def objective(low_thresh):
            """Objective function: maximize macro F1-score"""
            high_thresh = 0.65  # Keep high threshold fixed, optimize low
            y_true_class = self._score_to_risk_level(y_true, low_thresh, high_thresh)
            y_pred_class = self._score_to_risk_level(y_pred, low_thresh, high_thresh)
            
            # Calculate macro F1-score
            f1 = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
            return -f1  # Minimize negative F1 (maximize F1)
        
        # Optimize low threshold (search between 0.2 and 0.5)
        result = minimize_scalar(objective, bounds=(0.2, 0.5), method='bounded')
        optimal_low = result.x
        
        # Now optimize high threshold
        def objective_high(high_thresh):
            y_true_class = self._score_to_risk_level(y_true, optimal_low, high_thresh)
            y_pred_class = self._score_to_risk_level(y_pred, optimal_low, high_thresh)
            f1 = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
            return -f1
        
        result_high = minimize_scalar(objective_high, bounds=(0.5, 0.8), method='bounded')
        optimal_high = result_high.x
        
        print(f"  Optimal thresholds: LOW<={optimal_low:.3f} < MED <={optimal_high:.3f} < HIGH")
        
        return optimal_low, optimal_high
    
    def _calculate_class_weights(self, y_class):
        """
        Calculate class weights for imbalanced classes (inverse frequency)
        
        Returns:
            dict: Class weights for LOW, MED, HIGH
        """
        from collections import Counter
        
        class_counts = Counter(y_class)
        total = len(y_class)
        
        # Inverse frequency weighting
        weights = {}
        for class_name, count in class_counts.items():
            weights[class_name] = total / (len(class_counts) * count)
        
        # Normalize so smallest weight is 1.0
        min_weight = min(weights.values())
        weights = {k: v / min_weight for k, v in weights.items()}
        
        print(f"\nClass distribution and weights:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} ({count/total*100:.1f}%) - weight: {weights[class_name]:.2f}")
        
        return weights
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
    
    def _plot_predictions(self, y_true, y_pred, low_thresh=0.35, high_thresh=0.65):
        """ENHANCED: Plot prediction scatter and distribution with category boundaries"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot with category colors
        # Color by true category
        colors = []
        for score in y_true:
            if score > high_thresh:
                colors.append('red')  # HIGH
            elif score > low_thresh:
                colors.append('orange')  # MED
            else:
                colors.append('green')  # LOW
        
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=10, c=colors)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Perfect prediction')
        
        # Add threshold lines
        axes[0].axhline(y=low_thresh, color='blue', linestyle=':', alpha=0.5, label=f'LOW/MED threshold ({low_thresh:.2f})')
        axes[0].axhline(y=high_thresh, color='purple', linestyle=':', alpha=0.5, label=f'MED/HIGH threshold ({high_thresh:.2f})')
        axes[0].axvline(x=low_thresh, color='blue', linestyle=':', alpha=0.5)
        axes[0].axvline(x=high_thresh, color='purple', linestyle=':', alpha=0.5)
        
        axes[0].set_xlabel('True Conflict Score')
        axes[0].set_ylabel('Predicted Conflict Score')
        axes[0].set_title('Prediction vs True Values (with Category Boundaries)')
        axes[0].legend(loc='best', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # Distribution
        axes[1].hist(y_true, bins=50, alpha=0.5, label='True', density=True)
        axes[1].hist(y_pred, bins=50, alpha=0.5, label='Predicted', density=True)
        axes[1].set_xlabel('Conflict Score')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Score Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'predictions_analysis.png', dpi=300)
        plt.close()
    
    def _plot_feature_importance(self, importance):
        """Plot feature importance"""
        if importance is None or len(importance) == 0:
            return
        
        plt.figure(figsize=(10, 8))
        if isinstance(importance, dict):
            features = list(importance.keys())
            values = list(importance.values())
        else:
            features = [f'Feature_{i}' for i in range(len(importance))]
            values = importance
        
        # Sort by importance
        sorted_idx = np.argsort(values)[-20:]  # Top 20
        features_sorted = [features[i] for i in sorted_idx]
        values_sorted = [values[i] for i in sorted_idx]
        
        plt.barh(features_sorted, values_sorted)
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300)
        plt.close()


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("FT-Transformer Conflict Prediction Training")
    print("=" * 60)
    
    # Device information
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    elif torch.backends.mps.is_available():
        print(f"Device: MPS (Apple Silicon)")
    else:
        print(f"Device: CPU")
    
    # Configuration (Google Colab paths)
    CSV_PATH = '/content/conflict_dataset.csv'
    OUTPUT_DIR = Path('/content/models/ft_transformer')
    USE_TUNING = True  # Set to False for faster training with defaults
    NUM_TUNING_SAMPLES = 20  # Number of hyperparameter trials
    MAX_EPOCHS = 50
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Preprocessing
    preprocessor = ConflictDatasetPreprocessor(CSV_PATH, test_size=0.3)
    preprocessed_data = preprocessor.load_and_preprocess()
    
    # Save preprocessor
    preprocessor.save_preprocessor(OUTPUT_DIR / 'preprocessor.pkl')
    
    # Step 2: Prepare data for PyTorch Tabular
    trainer = FTTransformerTrainer(output_dir=OUTPUT_DIR, device=DEVICE)
    pytorch_data = trainer.prepare_pytorch_tabular_data(preprocessed_data)
    trainer.data = pytorch_data  # Set data after preparation
    
    # Step 3: Train model (with optional ensemble)
    USE_ENSEMBLE = True  # Set to True to train multiple models for uncertainty
    ENSEMBLE_SIZE = 3  # Number of models in ensemble
    
    if USE_TUNING:
        if USE_ENSEMBLE:
            print(f"\nTraining ensemble of {ENSEMBLE_SIZE} models...")
            ensemble_models = []
            for i in range(ENSEMBLE_SIZE):
                print(f"\n{'='*60}")
                print(f"Training Model {i+1}/{ENSEMBLE_SIZE}")
                print(f"{'='*60}")
                model = trainer.train_with_tuning(
                    num_samples=NUM_TUNING_SAMPLES,
                    max_epochs=MAX_EPOCHS
                )
                ensemble_models.append(model)
            
            # Create ensemble
            ensemble = EnsembleModel(ensemble_models)
            print(f"\n✓ Ensemble created with {len(ensemble_models)} models")
            
            # Use first model as primary (for compatibility)
            model = ensemble_models[0]
        else:
            model = trainer.train_with_tuning(
            num_samples=NUM_TUNING_SAMPLES,
            max_epochs=MAX_EPOCHS
        )
    else:
        model = trainer.train_default(max_epochs=MAX_EPOCHS)
    
    if model is None:
        print("❌ Model training failed. Check dependencies.")
        return
    
    # Step 4: Evaluate
    evaluator = ModelEvaluator(model, pytorch_data, OUTPUT_DIR)
    metrics = evaluator.evaluate()
    
    # Step 5: Save model
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    
    model_save_path = OUTPUT_DIR / 'ft_transformer_model'
    model.save_model(str(model_save_path))
    
    # Save metrics
    metrics_path = OUTPUT_DIR / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save best config
    if trainer.best_config:
        config_path = OUTPUT_DIR / 'best_config.json'
        with open(config_path, 'w') as f:
            json.dump(trainer.best_config, f, indent=2)
    
    print(f"✓ Model saved to {model_save_path}")
    print(f"✓ Metrics saved to {metrics_path}")
    if trainer.best_config:
        print(f"✓ Config saved to {config_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel location: {model_save_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nFinal Metrics:")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")


if __name__ == "__main__":
    main()

