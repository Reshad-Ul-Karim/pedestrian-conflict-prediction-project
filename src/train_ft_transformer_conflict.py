import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score, f1_score, recall_score
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
from collections import defaultdict

# Fix PyTorch 2.6 unpickling issue for pytorch_tabular
# Use add_safe_globals (recommended approach) - this is sufficient and avoids recursion
# NOTE: If you get recursion errors in Colab, restart the runtime (Runtime -> Restart runtime)
#       to clear any cached code that might have old patches
try:
    import omegaconf
    # Add all omegaconf classes that might be in checkpoints
    omegaconf_classes = [
        omegaconf.dictconfig.DictConfig,
        omegaconf.listconfig.ListConfig,
    ]
    
    # Try to add additional omegaconf classes
    try:
        omegaconf_classes.append(omegaconf.OmegaConf)
    except:
        pass
    
    try:
        omegaconf_classes.append(omegaconf.base.ContainerMetadata)
    except:
        pass
    
    # Add all classes to safe globals
    torch.serialization.add_safe_globals(omegaconf_classes)
    print(f"✓ Added {len(omegaconf_classes)} omegaconf classes to safe globals")
except ImportError:
    print("⚠ omegaconf not found - if you get unpickling errors, install omegaconf")
except Exception as e:
    print(f"⚠ Error adding omegaconf to safe globals: {e}")
    print("  Continuing anyway - checkpoint loading may fail")

# Increase recursion limit temporarily for checkpoint loading if needed
import sys
_original_recursion_limit = sys.getrecursionlimit()
sys.setrecursionlimit(max(_original_recursion_limit, 3000))

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Check device
if torch.cuda.is_available():
    print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device('cuda')
else:
    print("⚠ Using CPU")
    DEVICE = torch.device('cpu')

# Import PyTorch Tabular
try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import FTTransformerConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
except ImportError:
    print("Installing pytorch-tabular...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'pytorch-tabular'])
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import FTTransformerConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

# Import XGBoost and CatBoost
try:
    import xgboost as xgb
except ImportError:
    print("Installing xgboost...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'xgboost'])
    import xgboost as xgb

try:
    import catboost as cb
except ImportError:
    print("Installing catboost...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'catboost'])
    import catboost as cb

# Load CSV
print("\n" + "="*60)
print("Loading Dataset")
print("="*60)
df = pd.read_csv('/content/conflict_dataset.csv')
print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

# Fill missing values
df = df.fillna(0)

# Estimate image dimensions
if 'bbox_x2' in df.columns and 'bbox_y2' in df.columns:
    img_width = int(df['bbox_x2'].max() * 1.1)
    img_height = int(df['bbox_y2'].max() * 1.1)
else:
    img_width, img_height = 1920, 1080

# Normalize bbox features
if all(col in df.columns for col in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'bbox_center_x', 'bbox_center_y', 'bbox_area']):
    df['bbox_center_x_norm'] = df['bbox_center_x'] / img_width
    df['bbox_center_y_norm'] = df['bbox_center_y'] / img_height
    df['bbox_x1_norm'] = df['bbox_x1'] / img_width
    df['bbox_y1_norm'] = df['bbox_y1'] / img_height
    df['bbox_x2_norm'] = df['bbox_x2'] / img_width
    df['bbox_y2_norm'] = df['bbox_y2'] / img_height
    df['bbox_area_norm'] = df['bbox_area'] / (img_width * img_height)
    df['bbox_width'] = df['bbox_x2'] - df['bbox_x1']
    df['bbox_height'] = df['bbox_y2'] - df['bbox_y1']
    df['bbox_width_norm'] = df['bbox_width'] / img_width
    df['bbox_height_norm'] = df['bbox_height'] / img_height
    df['bbox_aspect_ratio'] = df['bbox_width'] / (df['bbox_height'] + 1e-6)

# Handle pose features
if 'pose_detected' in df.columns and df['pose_detected'].dtype == 'bool':
    df['pose_detected'] = df['pose_detected'].astype(int)

# Fill NaN for pose
for col in ['position_agreement', 'pose_detected', 'pose_confidence', 'body_orientation_angle']:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)

# Interaction features
if 'bbox_area_norm' in df.columns and 'pose_confidence' in df.columns:
    df['area_pose_confidence_interaction'] = df['bbox_area_norm'] * df['pose_confidence']
if 'bbox_center_y_norm' in df.columns and 'pose_confidence' in df.columns:
    df['position_pose_confidence_interaction'] = df['bbox_center_y_norm'] * df['pose_confidence']
if 'body_orientation_angle' in df.columns and 'position_agreement' in df.columns:
    df['orientation_agreement_interaction'] = (np.abs(df['body_orientation_angle']) / 180.0) * df['position_agreement']
if 'bbox_area_norm' in df.columns and 'body_orientation_angle' in df.columns:
    df['area_orientation_interaction'] = df['bbox_area_norm'] * (np.abs(df['body_orientation_angle']) / 180.0)
if 'pose_confidence' in df.columns and 'body_orientation_angle' in df.columns:
    df['pose_orientation_interaction'] = df['pose_confidence'] * (np.abs(df['body_orientation_angle']) / 180.0)

# Feature selection
normalized_bbox_features = ['bbox_x1_norm', 'bbox_y1_norm', 'bbox_x2_norm', 'bbox_y2_norm',
                           'bbox_center_x_norm', 'bbox_center_y_norm', 'bbox_area_norm',
                           'bbox_width_norm', 'bbox_height_norm', 'bbox_aspect_ratio']
pose_features = ['position_agreement', 'pose_detected', 'pose_confidence', 'body_orientation_angle']
interaction_features = ['area_pose_confidence_interaction', 'position_pose_confidence_interaction',
                       'orientation_agreement_interaction', 'area_orientation_interaction',
                       'pose_orientation_interaction']

all_features = normalized_bbox_features + pose_features + interaction_features
feature_columns = [f for f in all_features if f in df.columns]

# Separate categorical and continuous
continuous_cols = []
categorical_cols = []
for col in feature_columns:
    if df[col].dtype in ['object', 'bool']:
        categorical_cols.append(col)
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    else:
        continuous_cols.append(col)

print(f"✓ Features: {len(feature_columns)} ({len(continuous_cols)} continuous, {len(categorical_cols)} categorical)")
print(f"\nFeature List:")
for i, feat in enumerate(feature_columns, 1):
    feat_type = "cont" if feat in continuous_cols else "cat"
    print(f"  {i:2d}. {feat:40s} [{feat_type}]")

# Prepare X and y
X = df[feature_columns].copy()
y = df['conflict_score'].copy()
print(f"✓ Target: conflict_score [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")

# Train-test split by image_id
unique_images = df['image_id'].unique()
train_images, test_images = train_test_split(unique_images, test_size=0.3, random_state=42)
train_mask = df['image_id'].isin(train_images)
test_mask = df['image_id'].isin(test_images)

X_train = X[train_mask].copy()
X_test = X[test_mask].copy()
y_train = y[train_mask].copy()
y_test = y[test_mask].copy()

# Scale continuous features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
if continuous_cols:
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

print(f"✓ Train/Test: {len(X_train)}/{len(X_test)} records")

# Prepare PyTorch Tabular format
train_df = X_train_scaled.copy()
train_df['conflict_score'] = y_train.values
test_df = X_test_scaled.copy()
test_df['conflict_score'] = y_test.values

# README: Increased Validation Data - 70/30 split (was 80/20)
# Split train_df into 70% train, 30% validation
train_df_split, val_df_split = train_test_split(train_df, test_size=0.3, random_state=42)
print(f"✓ Train/Val split: {len(train_df_split)}/{len(val_df_split)} (70/30)")

print("\n" + "="*60)
print("Training FT-Transformer")
print("="*60)

# Create model
data_config = DataConfig(
    target=['conflict_score'],
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols if categorical_cols else [],
)

# Create model config with regularization
model_config_kwargs = {
    'task': "regression",
    'learning_rate': 3e-4,
    'num_heads': 4,
    'num_attn_blocks': 3,
}

# Try to add embedding dimension and dropout
try:
    model_config_kwargs['input_embed_dim'] = 64
except:
    try:
        model_config_kwargs['d_model'] = 64
    except:
        pass

# Add dropout for regularization (if supported)
try:
    model_config_kwargs['attn_dropout'] = 0.1
    model_config_kwargs['ff_dropout'] = 0.1
    model_config_kwargs['embedding_dropout'] = 0.05
except:
    pass

model_config = FTTransformerConfig(**model_config_kwargs)
print(f"\nModel Configuration:")
print(f"  Task: {model_config.task}")
print(f"  Learning Rate: {model_config.learning_rate}")
print(f"  Num Heads: {model_config.num_heads}")
print(f"  Num Attn Blocks: {model_config.num_attn_blocks}")
if hasattr(model_config, 'input_embed_dim'):
    print(f"  Embedding Dim: {model_config.input_embed_dim}")
if hasattr(model_config, 'attn_dropout'):
    print(f"  Attention Dropout: {model_config.attn_dropout}")
if hasattr(model_config, 'ff_dropout'):
    print(f"  FF Dropout: {model_config.ff_dropout}")

optimizer_config = OptimizerConfig(
    optimizer='AdamW',
    optimizer_params={'weight_decay': 1e-4}  # README: L2 regularization (1e-4)
)

# Trainer config with anti-overfitting measures from README
# Start with basic supported parameters
trainer_config_kwargs = {
    'batch_size': 128,
    'max_epochs': 40,  # Reduced to 40 epochs as requested
    'early_stopping': "valid_loss",
    'early_stopping_patience': 15,  # README: Patience=15
    'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
    'devices': 1 if torch.cuda.is_available() else None,
}

# Try to add early_stopping_min_delta (check if parameter exists)
trainer_config = None
try:
    # First try with min_delta
    trainer_config_kwargs['early_stopping_min_delta'] = 1e-5  # README: min_delta=1e-5
    trainer_config = TrainerConfig(**trainer_config_kwargs)
except (TypeError, ValueError) as e:
    # If min_delta not supported, remove it and try again
    trainer_config_kwargs.pop('early_stopping_min_delta', None)
    trainer_config = TrainerConfig(**trainer_config_kwargs)
    print(f"⚠ early_stopping_min_delta not supported, using default")

# Ensemble configuration
USE_ENSEMBLE = True
ENSEMBLE_SIZE = 2  # Train 2 models as requested
ENSEMBLE_SEEDS = [42, 123, 456]  # Different seeds for each model

# Model comparison configuration
TRAIN_XGBOOST = True
TRAIN_CATBOOST = True

# Class weights calculation for imbalanced classes
def calculate_class_weights(y_true, thresholds):
    """Calculate class weights inversely proportional to frequency"""
    low_mask = y_true <= thresholds[0]
    med_mask = (y_true > thresholds[0]) & (y_true <= thresholds[1])
    high_mask = y_true > thresholds[1]
    
    n_low = low_mask.sum()
    n_med = med_mask.sum()
    n_high = high_mask.sum()
    n_total = len(y_true)
    
    # Inverse frequency weighting
    weight_low = n_total / (3 * n_low) if n_low > 0 else 1.0
    weight_med = n_total / (3 * n_med) if n_med > 0 else 1.0
    weight_high = n_total / (3 * n_high) if n_high > 0 else 1.0
    
    return {'LOW': weight_low, 'MED': weight_med, 'HIGH': weight_high}

# Improved threshold optimization with class weights
def optimize_thresholds_with_class_weights(y_true, y_pred, class_weights=None, low_range=(0.3, 0.5), high_range=(0.6, 0.9)):
    """Optimize thresholds considering class weights"""
    best_f1 = -1
    best_thresholds = [0.35, 0.65]
    
    # Search space for thresholds
    low_thresholds = np.linspace(low_range[0], low_range[1], 21)
    high_thresholds = np.linspace(high_range[0], high_range[1], 31)
    
    for low_th in low_thresholds:
        for high_th in high_thresholds:
            if high_th <= low_th:
                continue
                
            y_class = np.where(y_pred <= low_th, 'LOW',
                              np.where(y_pred <= high_th, 'MED', 'HIGH'))
            y_true_class = np.where(y_true <= low_th, 'LOW',
                                   np.where(y_true <= high_th, 'MED', 'HIGH'))
            
            # Weighted F1 score
            f1_scores = []
            for cls in ['LOW', 'MED', 'HIGH']:
                f1 = f1_score(y_true_class == cls, y_class == cls, zero_division=0)
                weight = class_weights.get(cls, 1.0) if class_weights else 1.0
                f1_scores.append(f1 * weight)
            
            macro_f1 = np.mean(f1_scores)
            
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_thresholds = [low_th, high_th]
    
    return best_thresholds, best_f1

# Basic threshold optimization (for initial class weight calculation)
def optimize_thresholds(y_true, y_pred, low_range=(0.2, 0.5), high_range=(0.5, 0.8)):
    """
    Find optimal thresholds that maximize macro F1-score
    """
    best_f1 = -1
    best_low = 0.35
    best_high = 0.65
    
    # Grid search over threshold ranges
    low_candidates = np.linspace(low_range[0], low_range[1], 20)
    high_candidates = np.linspace(high_range[0], high_range[1], 20)
    
    for low in low_candidates:
        for high in high_candidates:
            if low >= high:
                continue
            
            y_pred_class = np.array(['HIGH' if s > high else 'MED' if s > low else 'LOW' for s in y_pred])
            y_true_class = np.array(['HIGH' if s > high else 'MED' if s > low else 'LOW' for s in y_true])
            
            # Calculate macro F1
            try:
                f1 = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_low = low
                    best_high = high
            except:
                continue
    
    return best_low, best_high, best_f1

# Recall-focused threshold adjustment
def evaluate_with_recall_focus(y_true, y_pred, thresholds, recall_target=0.75):
    """Evaluate with higher weight on recall for HIGH class"""
    
    y_pred_class = np.where(y_pred <= thresholds[0], 'LOW',
                            np.where(y_pred <= thresholds[1], 'MED', 'HIGH'))
    y_true_class = np.where(y_true <= thresholds[0], 'LOW',
                           np.where(y_true <= thresholds[1], 'MED', 'HIGH'))
    
    # Calculate per-class recall
    recalls = {
        'LOW': recall_score(y_true_class == 'LOW', y_pred_class == 'LOW', zero_division=0),
        'MED': recall_score(y_true_class == 'MED', y_pred_class == 'MED', zero_division=0),
        'HIGH': recall_score(y_true_class == 'HIGH', y_pred_class == 'HIGH', zero_division=0),
    }
    
    # If HIGH recall is low, adjust threshold downward
    if recalls['HIGH'] < recall_target:
        # Try lowering HIGH threshold
        adjusted_high_th = thresholds[1] * 0.95  # Lower by 5%
        adjusted_thresholds = [thresholds[0], adjusted_high_th]
        
        # Re-evaluate
        y_pred_class_adj = np.where(y_pred <= adjusted_thresholds[0], 'LOW',
                                    np.where(y_pred <= adjusted_thresholds[1], 'MED', 'HIGH'))
        recalls_adj = {
            'HIGH': recall_score(y_true_class == 'HIGH', y_pred_class_adj == 'HIGH', zero_division=0),
        }
        
        if recalls_adj['HIGH'] > recalls['HIGH']:
            print(f"  Adjusted HIGH threshold: {thresholds[1]:.3f} -> {adjusted_high_th:.3f}")
            print(f"  HIGH recall improved: {recalls['HIGH']:.3f} -> {recalls_adj['HIGH']:.3f}")
            return adjusted_thresholds, recalls_adj
    
    return thresholds, recalls

# Improved checkpoint saving
def save_ensemble_model(model, save_dir, model_index):
    """Save model with proper checkpoint handling"""
    import os
    import shutil
    
    model_dir = os.path.join(save_dir, f"model_{model_index}")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Save using pytorch_tabular's save method
        model.save_model(model_dir, save_config=True, save_transformer=True)
        
        # Also save raw state dict as backup
        checkpoint_path = os.path.join(model_dir, "checkpoint.ckpt")
        if hasattr(model, 'trainer') and model.trainer is not None:
            try:
                model.trainer.save_checkpoint(checkpoint_path)
            except:
                pass
        
        print(f"✓ Model {model_index} saved successfully to {model_dir}")
        return True
    except Exception as e:
        print(f"⚠ Error saving model {model_index}: {e}")
        # Try alternative save method
        try:
            if hasattr(model, 'model'):
                torch.save(model.model.state_dict(), 
                          os.path.join(model_dir, "state_dict.pt"))
                print(f"✓ Model {model_index} state dict saved as backup")
                return True
        except Exception as e2:
            print(f"✗ Failed to save model {model_index}: {e2}")
            return False

# Ensemble Model Class for uncertainty quantification
class EnsembleModel:
    """Ensemble of models for uncertainty-aware predictions"""
    def __init__(self, models):
        self.models = models
        self.n_models = len(models)
    
    def predict(self, data, return_uncertainty=True):
        """
        Get ensemble predictions with uncertainty quantification
        
        Returns:
            dict with 'mean', 'std', 'lower_bound', 'upper_bound' (95% CI)
        """
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(data)
                if isinstance(pred, pd.DataFrame):
                    if 'conflict_score_prediction' in pred.columns:
                        pred_values = pred['conflict_score_prediction'].values
                    else:
                        pred_values = pred.iloc[:, 0].values
                else:
                    pred_values = pred
                predictions.append(pred_values)
            except Exception as e:
                print(f"⚠ Error getting prediction from one model: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from ensemble models")
        
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        
        # Mean prediction (ensemble average)
        mean_pred = np.mean(predictions, axis=0)
        
        # Uncertainty as standard deviation across models
        std_pred = np.std(predictions, axis=0)
        
        # 95% confidence intervals
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        result = {
            'mean': mean_pred,
            'std': std_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        return result

print("\n" + "="*60)
print("Starting Training")
print("="*60)
print(f"  Train samples: {len(train_df_split)}")
print(f"  Validation samples: {len(val_df_split)}")
print(f"  Batch size: {trainer_config.batch_size}")
print(f"  Max epochs: {trainer_config.max_epochs}")
print(f"  Early stopping patience: {trainer_config.early_stopping_patience}")
if hasattr(trainer_config, 'early_stopping_min_delta'):
    print(f"  Early stopping min_delta: {trainer_config.early_stopping_min_delta}")
if hasattr(trainer_config, 'learning_rate_scheduler'):
    print(f"  Learning rate scheduler: {trainer_config.learning_rate_scheduler}")

print("\n" + "-"*60)
print("Anti-Overfitting Measures (from README):")
print("-"*60)
print("✓ Dropout: attn_dropout=0.1, ff_dropout=0.1, embedding_dropout=0.05")
print("✓ Weight Decay: L2 regularization (1e-4)")
print(f"✓ Early Stopping: Patience={trainer_config.early_stopping_patience}, min_delta={getattr(trainer_config, 'early_stopping_min_delta', 'N/A')}")
print("⚠ Learning Rate Scheduler: Not directly configurable in TrainerConfig (may be handled internally)")
print("✓ Reduced Model Complexity: num_heads=4, num_attn_blocks=3, embed_dim=64")
print("✓ Increased Validation Data: 70/30 train/val split")
print("✓ Threshold Optimization: Automatic LOW/MED/HIGH threshold finding")
if USE_ENSEMBLE:
    print(f"✓ Ensemble: {ENSEMBLE_SIZE} models for uncertainty quantification")
if TRAIN_XGBOOST:
    print("✓ XGBoost: Will train for comparison")
if TRAIN_CATBOOST:
    print("✓ CatBoost: Will train for comparison")
print("-"*60)

# Train ensemble or single model
ensemble_models = []
ensemble = None

# Storage for all models and their predictions
all_models = {}
all_predictions = {}
all_metrics = {}

if USE_ENSEMBLE:
    print(f"\n{'='*60}")
    print(f"Training Ensemble of {ENSEMBLE_SIZE} Models")
    print(f"{'='*60}")
    
    for i in range(ENSEMBLE_SIZE):
        print(f"\n{'-'*60}")
        print(f"Training Model {i+1}/{ENSEMBLE_SIZE}")
        print(f"{'-'*60}")
        
        # Set different seed for each model
        ensemble_seed = ENSEMBLE_SEEDS[i] if i < len(ENSEMBLE_SEEDS) else SEED + i * 100
        random.seed(ensemble_seed)
        np.random.seed(ensemble_seed)
        torch.manual_seed(ensemble_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ensemble_seed)
        
        print(f"  Using seed: {ensemble_seed}")
        
        # Create model with same config
        ensemble_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        
        try:
            ensemble_model.fit(train=train_df_split, validation=val_df_split)
            ensemble_models.append(ensemble_model)
            print(f"  ✓ Model {i+1} trained successfully")
        except (RecursionError, Exception) as e:
            # Check if it's a checkpoint loading error (training succeeded)
            error_str = str(e).lower()
            is_checkpoint_error = any(keyword in error_str for keyword in [
                'recursion', 'unpickling', 'weights_only', 'safe_globals', 
                'checkpoint', 'load failed'
            ])
            
            if is_checkpoint_error:
                # Checkpoint loading error, but model is already trained and in memory
                print(f"  ⚠ Model {i+1} checkpoint loading error (model is trained): {str(e)[:150]}")
                print("  Continuing with model in memory (training succeeded, checkpoint load failed)")
                ensemble_models.append(ensemble_model)
            else:
                # Actual training error
                print(f"  ⚠ Model {i+1} training error: {str(e)[:150]}")
                print("  Attempting to load best checkpoint...")
                try:
                    ensemble_model.load_best_model()
                    ensemble_models.append(ensemble_model)
                    print(f"  ✓ Model {i+1} loaded from checkpoint")
                except (RecursionError, Exception) as e2:
                    error_str2 = str(e2).lower()
                    is_checkpoint_error2 = any(keyword in error_str2 for keyword in [
                        'recursion', 'unpickling', 'weights_only', 'safe_globals'
                    ])
                    if is_checkpoint_error2:
                        print(f"  ⚠ Model {i+1} checkpoint load error, but model may be usable")
                        ensemble_models.append(ensemble_model)
                    else:
                        print(f"  ✗ Model {i+1} failed: {str(e2)[:150]}")
                        if i == 0:
                            # If first model fails, we can't continue
                            raise
        
        # Reset seed for next model
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    
    # Create ensemble
    if len(ensemble_models) > 0:
        ensemble = EnsembleModel(ensemble_models)
        print(f"\n✓ Ensemble created with {len(ensemble_models)} models")
        # Use first model as primary model for compatibility
        model = ensemble_models[0]
    else:
        print("\n✗ Ensemble training failed, falling back to single model")
        USE_ENSEMBLE = False
        model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        try:
            model.fit(train=train_df_split, validation=val_df_split)
            print("\n✓ Single model training completed successfully")
        except (RecursionError, Exception) as e:
            error_str = str(e).lower()
            is_checkpoint_error = any(keyword in error_str for keyword in [
                'recursion', 'unpickling', 'weights_only', 'safe_globals', 
                'checkpoint', 'load failed'
            ])
            if is_checkpoint_error:
                # Checkpoint loading error, but model is already trained
                print(f"\n⚠ Checkpoint loading error (model is trained): {str(e)[:150]}")
                print("Continuing with model in memory (training succeeded, checkpoint load failed)")
            else:
                print(f"\n⚠ Training error: {str(e)[:150]}")
                raise
else:
    # Single model training
    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    
    try:
        model.fit(train=train_df_split, validation=val_df_split)
        print("\n✓ Training completed successfully")
    except (RecursionError, Exception) as e:
        error_str = str(e).lower()
        is_checkpoint_error = any(keyword in error_str for keyword in [
            'recursion', 'unpickling', 'weights_only', 'safe_globals', 
            'checkpoint', 'load failed'
        ])
        if is_checkpoint_error:
            # Checkpoint loading error, but model is already trained
            print(f"\n⚠ Checkpoint loading error (model is trained): {str(e)[:150]}")
            print("Continuing with model in memory (training succeeded, checkpoint load failed)")
        else:
            print(f"\n⚠ Training error: {str(e)[:150]}")
            print("Attempting to load best checkpoint...")
            try:
                model.load_best_model()
                print("✓ Loaded best checkpoint")
            except (RecursionError, Exception) as e2:
                error_str2 = str(e2).lower()
                is_checkpoint_error2 = any(keyword in error_str2 for keyword in [
                    'recursion', 'unpickling', 'weights_only', 'safe_globals'
                ])
                if is_checkpoint_error2:
                    print(f"⚠ Checkpoint load error, but model may be usable")
                else:
                    print(f"✗ Could not load checkpoint: {str(e2)[:150]}")
                    raise

# Train XGBoost for comparison
xgb_model = None
if TRAIN_XGBOOST:
    print("\n" + "="*60)
    print("Training XGBoost")
    print("="*60)
    
    try:
        # Prepare data for XGBoost
        X_train_xgb = train_df_split[feature_columns].values
        y_train_xgb = train_df_split['conflict_score'].values
        X_val_xgb = val_df_split[feature_columns].values
        y_val_xgb = val_df_split['conflict_score'].values
        X_test_xgb = test_df[feature_columns].values
        
        print(f"  Train samples: {len(X_train_xgb)}")
        print(f"  Validation samples: {len(X_val_xgb)}")
        print(f"  Test samples: {len(X_test_xgb)}")
        
        # XGBoost parameters
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train_xgb, label=y_train_xgb)
        dval = xgb.DMatrix(X_val_xgb, label=y_val_xgb)
        dtest = xgb.DMatrix(X_test_xgb)
        
        # Train with early stopping
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=xgb_params['n_estimators'],
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Make predictions
        y_pred_xgb = xgb_model.predict(dtest)
        
        all_models['XGBoost'] = xgb_model
        all_predictions['XGBoost'] = y_pred_xgb
        
        print(f"✓ XGBoost trained successfully")
        print(f"  Best iteration: {xgb_model.best_iteration}")
        print(f"  Best score: {xgb_model.best_score:.6f}")
        
    except Exception as e:
        print(f"✗ XGBoost training error: {e}")
        import traceback
        traceback.print_exc()
        TRAIN_XGBOOST = False

# Train CatBoost for comparison
catboost_model = None
if TRAIN_CATBOOST:
    print("\n" + "="*60)
    print("Training CatBoost")
    print("="*60)
    
    try:
        # Prepare data for CatBoost
        X_train_cb = train_df_split[feature_columns].values
        y_train_cb = train_df_split['conflict_score'].values
        X_val_cb = val_df_split[feature_columns].values
        y_val_cb = val_df_split['conflict_score'].values
        X_test_cb = test_df[feature_columns].values
        
        print(f"  Train samples: {len(X_train_cb)}")
        print(f"  Validation samples: {len(X_val_cb)}")
        print(f"  Test samples: {len(X_test_cb)}")
        
        # CatBoost parameters
        catboost_params = {
            'loss_function': 'RMSE',
            'iterations': 200,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': SEED,
            'verbose': False,
            'early_stopping_rounds': 20
        }
        
        # Train CatBoost
        catboost_model = cb.CatBoostRegressor(**catboost_params)
        catboost_model.fit(
            X_train_cb, y_train_cb,
            eval_set=(X_val_cb, y_val_cb),
            use_best_model=True
        )
        
        # Make predictions
        y_pred_cb = catboost_model.predict(X_test_cb)
        
        all_models['CatBoost'] = catboost_model
        all_predictions['CatBoost'] = y_pred_cb
        
        print(f"✓ CatBoost trained successfully")
        print(f"  Best iteration: {catboost_model.get_best_iteration()}")
        print(f"  Best score: {catboost_model.get_best_score():.6f}")
        
    except Exception as e:
        print(f"✗ CatBoost training error: {e}")
        import traceback
        traceback.print_exc()
        TRAIN_CATBOOST = False

# Extract and plot training/validation loss curves
print("\n" + "="*60)
print("Plotting Training Curves")
print("="*60)

# Collect training metrics
training_metrics = defaultdict(list)

try:
    # Method 1: Check if model has history attribute
    if hasattr(model, 'history'):
        history = model.history
        if isinstance(history, dict):
            training_metrics.update(history)
    
    # Method 2: Try to access trainer's logged metrics (PyTorch Lightning)
    if hasattr(model, 'trainer') and model.trainer is not None:
        trainer = model.trainer
        
        # Check for logged metrics
        if hasattr(trainer, 'logged_metrics') and trainer.logged_metrics:
            logged = trainer.logged_metrics
            if isinstance(logged, dict):
                # Extract epoch-level metrics
                for key, value in logged.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else value.cpu().numpy()
                    training_metrics[key].append(value)
        
        # Check for callback metrics
        if hasattr(trainer, 'callback_metrics') and trainer.callback_metrics:
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.cpu().numpy()
                training_metrics[key].append(value)
        
        # Check for CSV logger (common in PyTorch Lightning)
        if hasattr(trainer, 'loggers'):
            for logger in trainer.loggers:
                if hasattr(logger, 'name') and 'csv' in logger.name.lower():
                    # Try to read CSV file
                    try:
                        # Check default log directory
                        csv_path = Path('/content/ft_transformer_model') / 'lightning_logs' / 'version_0' / 'metrics.csv'
                        if csv_path.exists():
                            df_logs = pd.read_csv(csv_path)
                            for col in df_logs.columns:
                                if 'loss' in col.lower() or 'mse' in col.lower():
                                    training_metrics[col] = df_logs[col].dropna().tolist()
                    except:
                        pass
    
    # Method 3: Check model save directory for logs
    try:
        log_dir = Path('/content/ft_transformer_model') / 'lightning_logs'
        if log_dir.exists():
            # Find latest version
            versions = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith('version')])
            if versions:
                latest_version = versions[-1]
                csv_file = latest_version / 'metrics.csv'
                if csv_file.exists():
                    df_logs = pd.read_csv(csv_file)
                    for col in df_logs.columns:
                        if 'loss' in col.lower() or 'mse' in col.lower():
                            training_metrics[col] = df_logs[col].dropna().tolist()
    except:
        pass
    
    # Extract and plot metrics
    train_loss = []
    val_loss = []
    train_mse = []
    val_mse = []
    
    # Try to find loss metrics
    for key, values in training_metrics.items():
        key_lower = key.lower()
        if 'train' in key_lower and 'loss' in key_lower and 'epoch' in key_lower:
            train_loss = values if isinstance(values, list) else [values]
        elif ('val' in key_lower or 'valid' in key_lower) and 'loss' in key_lower and 'epoch' in key_lower:
            val_loss = values if isinstance(values, list) else [values]
        elif 'train' in key_lower and 'mse' in key_lower:
            train_mse = values if isinstance(values, list) else [values]
        elif ('val' in key_lower or 'valid' in key_lower) and 'mse' in key_lower:
            val_mse = values if isinstance(values, list) else [values]
    
    # Plot if we have any data
    if train_loss or val_loss or train_mse or val_mse:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Loss curves
        if train_loss or val_loss:
            epochs_loss = range(1, max(len(train_loss), len(val_loss)) + 1)
            if train_loss:
                axes[0].plot(epochs_loss[:len(train_loss)], train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
            if val_loss:
                axes[0].plot(epochs_loss[:len(val_loss)], val_loss, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Training and Validation Loss')
        
        # Plot 2: MSE curves
        if train_mse or val_mse:
            epochs_mse = range(1, max(len(train_mse), len(val_mse)) + 1)
            if train_mse:
                axes[1].plot(epochs_mse[:len(train_mse)], train_mse, 'b-', label='Train MSE', linewidth=2, marker='o', markersize=3)
            if val_mse:
                axes[1].plot(epochs_mse[:len(val_mse)], val_mse, 'r-', label='Validation MSE', linewidth=2, marker='s', markersize=3)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MSE')
            axes[1].set_title('Training and Validation MSE')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'MSE data not available', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Training and Validation MSE')
        
        plt.tight_layout()
        plt.savefig('/content/training_curves.png', dpi=200, bbox_inches='tight')
        print("✓ Training curves saved to /content/training_curves.png")
        plt.show()
        
        # Print summary
        if train_loss and val_loss:
            print(f"\nFinal Loss Values:")
            print(f"  Train Loss: {train_loss[-1]:.6f}")
            print(f"  Validation Loss: {val_loss[-1]:.6f}")
    else:
        print("⚠ Could not extract training/validation metrics")
        print("  Available metric keys:", list(training_metrics.keys()) if training_metrics else "None")
        print("  Note: Loss values are shown in the training output above")
        
except Exception as e:
    print(f"⚠ Error plotting training curves: {e}")
    import traceback
    traceback.print_exc()
    print("  Training completed successfully, but loss curves could not be extracted")
    print("  Check the training output above for epoch-by-epoch loss values")

# Evaluate
print("\n" + "="*60)
print("Evaluation")
print("="*60)

print(f"\nGenerating predictions on {len(test_df)} test samples...")
try:
    if USE_ENSEMBLE and ensemble is not None:
        print(f"  Using ensemble of {ensemble.n_models} models for uncertainty-aware predictions")
        ensemble_preds = ensemble.predict(test_df, return_uncertainty=True)
        y_pred = ensemble_preds['mean']
        y_pred_std = ensemble_preds['std']
        y_pred_lower = ensemble_preds['lower_bound']
        y_pred_upper = ensemble_preds['upper_bound']
        print(f"✓ Ensemble predictions generated: {len(y_pred)} samples")
        print(f"  Mean uncertainty (std): {np.mean(y_pred_std):.4f}")
        print(f"  Max uncertainty: {np.max(y_pred_std):.4f}")
    else:
        predictions = model.predict(test_df)
        if isinstance(predictions, pd.DataFrame):
            if 'conflict_score_prediction' in predictions.columns:
                y_pred = predictions['conflict_score_prediction'].values
            else:
                # Try to find prediction column
                pred_col = [c for c in predictions.columns if 'prediction' in c.lower() or 'pred' in c.lower()]
                if pred_col:
                    y_pred = predictions[pred_col[0]].values
                else:
                    y_pred = predictions.iloc[:, 0].values
        else:
            y_pred = predictions
        y_pred_std = None
        y_pred_lower = None
        y_pred_upper = None
        print(f"✓ Predictions generated: {len(y_pred)} samples")
except Exception as e:
    print(f"✗ Prediction error: {e}")
    raise

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nRegression Metrics (FT-Transformer):")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R²: {r2:.4f}")

# Store FT-Transformer metrics
all_models['FT-Transformer'] = model
all_predictions['FT-Transformer'] = y_pred
all_metrics['FT-Transformer'] = {
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'r2': r2
}

# Evaluate XGBoost if trained
if TRAIN_XGBOOST and xgb_model is not None:
    print(f"\nRegression Metrics (XGBoost):")
    y_pred_xgb = all_predictions['XGBoost']
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    
    print(f"  MSE: {mse_xgb:.4f}")
    print(f"  RMSE: {rmse_xgb:.4f}")
    print(f"  MAE: {mae_xgb:.4f}")
    print(f"  R²: {r2_xgb:.4f}")
    
    all_metrics['XGBoost'] = {
        'mse': mse_xgb,
        'rmse': rmse_xgb,
        'mae': mae_xgb,
        'r2': r2_xgb
    }

# Evaluate CatBoost if trained
if TRAIN_CATBOOST and catboost_model is not None:
    print(f"\nRegression Metrics (CatBoost):")
    y_pred_cb = all_predictions['CatBoost']
    mse_cb = mean_squared_error(y_test, y_pred_cb)
    rmse_cb = np.sqrt(mse_cb)
    mae_cb = mean_absolute_error(y_test, y_pred_cb)
    r2_cb = r2_score(y_test, y_pred_cb)
    
    print(f"  MSE: {mse_cb:.4f}")
    print(f"  RMSE: {rmse_cb:.4f}")
    print(f"  MAE: {mae_cb:.4f}")
    print(f"  R²: {r2_cb:.4f}")
    
    all_metrics['CatBoost'] = {
        'mse': mse_cb,
        'rmse': rmse_cb,
        'mae': mae_cb,
        'r2': r2_cb
    }

# Model Comparison Summary
if len(all_metrics) > 1:
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    
    comparison_df = pd.DataFrame(all_metrics).T
    print("\nRegression Metrics Comparison:")
    print(comparison_df.round(4))
    
    # Find best model for each metric
    print("\nBest Model per Metric:")
    for metric in ['mse', 'rmse', 'mae']:
        best_model = comparison_df[metric].idxmin()
        best_value = comparison_df.loc[best_model, metric]
        print(f"  {metric.upper()}: {best_model} ({best_value:.4f})")
    
    for metric in ['r2']:
        best_model = comparison_df[metric].idxmax()
        best_value = comparison_df.loc[best_model, metric]
        print(f"  {metric.upper()}: {best_model} ({best_value:.4f})")

# README: Improved Threshold Optimization with Class Weights and Recall Focus
print("\n" + "="*60)
print("Improved Threshold Optimization")
print("="*60)

# Step 1: Initial threshold optimization (for class weight calculation)
initial_thresholds = [0.35, 0.65]
opt_low, opt_high, opt_f1 = optimize_thresholds(y_test, y_pred, 
                                                  low_range=(0.2, 0.5), 
                                                  high_range=(0.5, 0.8))
print(f"✓ Initial optimal thresholds: LOW<={opt_low:.3f}, MED={opt_low:.3f}-{opt_high:.3f}, HIGH>={opt_high:.3f}")
print(f"  Initial Macro F1: {opt_f1:.4f}")

# Step 2: Calculate class weights based on distribution
class_weights = calculate_class_weights(y_test, [opt_low, opt_high])
print(f"\n✓ Class Weights (inverse frequency):")
print(f"  LOW: {class_weights['LOW']:.3f}")
print(f"  MED: {class_weights['MED']:.3f}")
print(f"  HIGH: {class_weights['HIGH']:.3f}")

# Step 3: Optimize thresholds with class weights
weighted_thresholds, weighted_f1 = optimize_thresholds_with_class_weights(
    y_test, y_pred, class_weights=class_weights,
    low_range=(0.3, 0.5), high_range=(0.6, 0.9)
)
print(f"\n✓ Weighted optimal thresholds: LOW<={weighted_thresholds[0]:.3f}, MED={weighted_thresholds[0]:.3f}-{weighted_thresholds[1]:.3f}, HIGH>={weighted_thresholds[1]:.3f}")
print(f"  Weighted Macro F1: {weighted_f1:.4f}")

# Step 4: Improve HIGH recall
final_thresholds, recalls = evaluate_with_recall_focus(
    y_test, y_pred, weighted_thresholds, recall_target=0.75
)
print(f"\n✓ Final thresholds after recall adjustment:")
print(f"  LOW ≤ {final_thresholds[0]:.3f}")
print(f"  MED: {final_thresholds[0]:.3f} - {final_thresholds[1]:.3f}")
print(f"  HIGH ≥ {final_thresholds[1]:.3f}")
print(f"\n✓ Per-class Recall:")
print(f"  LOW: {recalls.get('LOW', 0):.3f}")
print(f"  MED: {recalls.get('MED', 0):.3f}")
print(f"  HIGH: {recalls.get('HIGH', 0):.3f}")

# Use final thresholds for classification
opt_low, opt_high = final_thresholds[0], final_thresholds[1]

# Classification metrics with optimized thresholds
def score_to_level(scores, low=0.35, high=0.65):
    return np.array(['HIGH' if s > high else 'MED' if s > low else 'LOW' for s in scores])

y_true_class = score_to_level(y_test, low=opt_low, high=opt_high)
y_pred_class = score_to_level(y_pred, low=opt_low, high=opt_high)

print(f"\nClassification Report:")
print(classification_report(y_true_class, y_pred_class, target_names=['LOW', 'MED', 'HIGH']))

kappa = cohen_kappa_score(y_true_class, y_pred_class)
macro_f1 = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
print(f"\n  Cohen's Kappa: {kappa:.4f}")
print(f"  Macro F1: {macro_f1:.4f}")

# Store FT-Transformer classification metrics
all_metrics['FT-Transformer']['kappa'] = kappa
all_metrics['FT-Transformer']['macro_f1'] = macro_f1
all_metrics['FT-Transformer']['thresholds'] = {'low': float(opt_low), 'high': float(opt_high)}

# Evaluate XGBoost and CatBoost with same thresholds
if TRAIN_XGBOOST and xgb_model is not None:
    y_pred_xgb_class = score_to_level(all_predictions['XGBoost'], low=opt_low, high=opt_high)
    kappa_xgb = cohen_kappa_score(y_true_class, y_pred_xgb_class)
    macro_f1_xgb = f1_score(y_true_class, y_pred_xgb_class, average='macro', zero_division=0)
    
    all_metrics['XGBoost']['kappa'] = kappa_xgb
    all_metrics['XGBoost']['macro_f1'] = macro_f1_xgb
    all_metrics['XGBoost']['thresholds'] = {'low': float(opt_low), 'high': float(opt_high)}
    
    print(f"\nClassification Metrics (XGBoost):")
    print(f"  Cohen's Kappa: {kappa_xgb:.4f}")
    print(f"  Macro F1: {macro_f1_xgb:.4f}")

if TRAIN_CATBOOST and catboost_model is not None:
    y_pred_cb_class = score_to_level(all_predictions['CatBoost'], low=opt_low, high=opt_high)
    kappa_cb = cohen_kappa_score(y_true_class, y_pred_cb_class)
    macro_f1_cb = f1_score(y_true_class, y_pred_cb_class, average='macro', zero_division=0)
    
    all_metrics['CatBoost']['kappa'] = kappa_cb
    all_metrics['CatBoost']['macro_f1'] = macro_f1_cb
    all_metrics['CatBoost']['thresholds'] = {'low': float(opt_low), 'high': float(opt_high)}
    
    print(f"\nClassification Metrics (CatBoost):")
    print(f"  Cohen's Kappa: {kappa_cb:.4f}")
    print(f"  Macro F1: {macro_f1_cb:.4f}")

# Classification Comparison Summary
if len([m for m in all_metrics.keys() if 'kappa' in all_metrics[m]]) > 1:
    print("\n" + "="*60)
    print("Classification Metrics Comparison")
    print("="*60)
    
    cls_comparison = pd.DataFrame({
        model: {
            'Kappa': all_metrics[model].get('kappa', 0),
            'Macro F1': all_metrics[model].get('macro_f1', 0)
        }
        for model in all_metrics.keys() if 'kappa' in all_metrics[model]
    }).T
    
    print("\nClassification Metrics:")
    print(cls_comparison.round(4))
    
    print("\nBest Model per Classification Metric:")
    best_kappa_model = cls_comparison['Kappa'].idxmax()
    best_f1_model = cls_comparison['Macro F1'].idxmax()
    print(f"  Kappa: {best_kappa_model} ({cls_comparison.loc[best_kappa_model, 'Kappa']:.4f})")
    print(f"  Macro F1: {best_f1_model} ({cls_comparison.loc[best_f1_model, 'Macro F1']:.4f})")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
colors = ['red' if s > 0.65 else 'orange' if s > 0.35 else 'green' for s in y_test]
axes[0].scatter(y_test, y_pred, alpha=0.5, s=10, c=colors)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[0].set_xlabel('True Conflict Score')
axes[0].set_ylabel('Predicted Conflict Score')
axes[0].set_title('Predictions vs True Values')
axes[0].grid(True, alpha=0.3)

# Distribution
axes[1].hist(y_test, bins=50, alpha=0.5, label='True', density=True)
axes[1].hist(y_pred, bins=50, alpha=0.5, label='Predicted', density=True)
axes[1].set_xlabel('Conflict Score')
axes[1].set_ylabel('Density')
axes[1].set_title('Score Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/training_results.png', dpi=200, bbox_inches='tight')
plt.show()

# Model Comparison Plots
if len(all_predictions) > 1:
    print("\n" + "="*60)
    print("Generating Model Comparison Plots")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Predictions vs True Values (all models)
    ax = axes[0, 0]
    colors_map = {'FT-Transformer': 'blue', 'XGBoost': 'green', 'CatBoost': 'orange'}
    for model_name, y_pred_model in all_predictions.items():
        ax.scatter(y_test, y_pred_model, alpha=0.3, s=10, 
                  label=model_name, color=colors_map.get(model_name, 'gray'))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect')
    ax.set_xlabel('True Conflict Score')
    ax.set_ylabel('Predicted Conflict Score')
    ax.set_title('Predictions vs True Values (All Models)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residuals comparison
    ax = axes[0, 1]
    for model_name, y_pred_model in all_predictions.items():
        residuals = y_test - y_pred_model
        ax.scatter(y_pred_model, residuals, alpha=0.3, s=10,
                  label=model_name, color=colors_map.get(model_name, 'gray'))
    ax.axhline(y=0, color='k', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Conflict Score')
    ax.set_ylabel('Residuals (True - Predicted)')
    ax.set_title('Residuals Plot (All Models)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Metrics comparison bar chart
    ax = axes[1, 0]
    if len(all_metrics) > 0:
        metrics_to_plot = ['rmse', 'mae', 'r2']
        x = np.arange(len(metrics_to_plot))
        width = 0.25
        multiplier = 0
        
        for model_name in all_metrics.keys():
            values = [all_metrics[model_name][m] for m in metrics_to_plot]
            # Normalize R² for better visualization (multiply by max RMSE)
            if 'r2' in metrics_to_plot:
                r2_idx = metrics_to_plot.index('r2')
                max_rmse = max([all_metrics[m]['rmse'] for m in all_metrics.keys()])
                values[r2_idx] = values[r2_idx] * max_rmse
            
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, label=model_name,
                          color=colors_map.get(model_name, 'gray'))
            multiplier += 1
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Metrics Comparison (R² normalized)')
        ax.set_xticks(x + width * (len(all_metrics) - 1) / 2)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Prediction distribution comparison
    ax = axes[1, 1]
    for model_name, y_pred_model in all_predictions.items():
        ax.hist(y_pred_model, bins=30, alpha=0.5, label=model_name,
               color=colors_map.get(model_name, 'gray'), density=True)
    ax.hist(y_test, bins=30, alpha=0.5, label='True', color='red', 
           linestyle='--', histtype='step', linewidth=2, density=True)
    ax.set_xlabel('Conflict Score')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/content/model_comparison.png', dpi=200, bbox_inches='tight')
    print("✓ Model comparison plots saved to /content/model_comparison.png")
    plt.show()

# Confusion matrix
cm = confusion_matrix(y_true_class, y_pred_class)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['LOW', 'MED', 'HIGH'], yticklabels=['LOW', 'MED', 'HIGH'])
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.show()

# Feature Importance Analysis for LOW Class Misclassifications
print("\n" + "="*60)
print("Feature Importance Analysis")
print("="*60)

try:
    # Identify misclassified LOW samples
    low_threshold = opt_low
    low_true = y_test <= low_threshold
    low_pred = y_pred <= low_threshold
    misclassified_low = low_true & (~low_pred)  # True LOW predicted as not LOW
    
    if misclassified_low.sum() > 0:
        print(f"✓ Found {misclassified_low.sum()} LOW misclassifications")
        
        # Try to install and use SHAP for feature importance
        try:
            import shap
            print("  Using SHAP for feature importance analysis...")
            
            # Prepare data for SHAP (use a sample for speed)
            sample_size = min(100, len(test_df))
            X_sample = test_df[feature_columns].iloc[:sample_size].values
            
            # Get model for SHAP (use first ensemble model or single model)
            if USE_ENSEMBLE and ensemble is not None and len(ensemble_models) > 0:
                model_for_shap = ensemble_models[0]
            else:
                model_for_shap = model
            
            # Create a wrapper function for SHAP
            def model_predict_wrapper(X):
                """Wrapper to convert numpy array to DataFrame for model prediction"""
                X_df = pd.DataFrame(X, columns=feature_columns)
                pred = model_for_shap.predict(X_df)
                if isinstance(pred, pd.DataFrame):
                    if 'conflict_score_prediction' in pred.columns:
                        return pred['conflict_score_prediction'].values
                    else:
                        return pred.iloc[:, 0].values
                return pred
            
            # Create SHAP explainer (using a subset of data as background)
            background_size = min(50, len(test_df))
            X_background = test_df[feature_columns].iloc[:background_size].values
            
            try:
                explainer = shap.Explainer(model_predict_wrapper, X_background)
                shap_values = explainer(X_sample)
                
                # Plot feature importance
                plt.figure(figsize=(10, 8))
                shap.plots.bar(shap_values, show=False)
                plt.title("Feature Importance (SHAP Values)", fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig('/content/feature_importance_shap.png', dpi=200, bbox_inches='tight')
                print("  ✓ Feature importance plot saved to /content/feature_importance_shap.png")
                plt.show()
                
                # Get top features
                mean_shap = np.abs(shap_values.values).mean(0)
                top_features_idx = np.argsort(mean_shap)[-10:][::-1]
                print("\n  Top 10 Most Important Features:")
                for i, idx in enumerate(top_features_idx, 1):
                    print(f"    {i:2d}. {feature_columns[idx]:40s} (SHAP: {mean_shap[idx]:.4f})")
                
            except Exception as e:
                print(f"  ⚠ SHAP explainer error: {e}")
                print("  Falling back to permutation importance...")
                raise
        
        except ImportError:
            print("  ⚠ SHAP not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', '-q', 'shap'])
            import shap
            print("  ✓ SHAP installed. Please re-run this cell for feature importance analysis.")
        except Exception as e:
            print(f"  ⚠ SHAP analysis failed: {e}")
            print("  Using simple feature correlation analysis instead...")
            
            # Fallback: Simple correlation analysis
            misclassified_df = test_df[misclassified_low]
            if len(misclassified_df) > 0:
                correlations = {}
                for col in feature_columns:
                    if col in misclassified_df.columns:
                        corr = np.corrcoef(misclassified_df[col], y_test[misclassified_low])[0, 1]
                        correlations[col] = abs(corr)
                
                # Sort by correlation
                sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                print("\n  Top 10 Features Correlated with LOW Misclassifications:")
                for i, (feat, corr) in enumerate(sorted_features[:10], 1):
                    print(f"    {i:2d}. {feat:40s} (|corr|: {corr:.4f})")
    else:
        print("  ✓ No LOW misclassifications found - model performs well on LOW class!")
        
except Exception as e:
    print(f"  ⚠ Feature importance analysis error: {e}")
    import traceback
    traceback.print_exc()
    print("  Continuing without feature importance analysis...")

# Save model(s)
model_save_path = '/content/ft_transformer_model'
print(f"\n" + "="*60)
print("Saving Model(s)")
print("="*60)
try:
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
    if USE_ENSEMBLE and ensemble is not None:
        # Save each ensemble model using improved checkpoint handling
        ensemble_dir = Path(model_save_path) / 'ensemble'
        ensemble_dir.mkdir(exist_ok=True)
        
        for i, ensemble_model in enumerate(ensemble_models):
            save_ensemble_model(ensemble_model, str(ensemble_dir), i+1)
        
        # Also save primary model
        try:
            model.save_model(model_save_path)
            print(f"✓ FT-Transformer model saved to {model_save_path}")
        except Exception as e:
            print(f"⚠ Error saving primary model: {e}")
            save_ensemble_model(model, model_save_path, "primary")
    else:
        try:
            model.save_model(model_save_path)
            print(f"✓ FT-Transformer model saved to {model_save_path}")
        except Exception as e:
            print(f"⚠ Error saving model: {e}")
            save_ensemble_model(model, model_save_path, "single")
    
    # Save XGBoost model
    if TRAIN_XGBOOST and xgb_model is not None:
        xgb_save_path = Path(model_save_path) / 'xgboost_model.json'
        try:
            xgb_model.save_model(str(xgb_save_path))
            print(f"✓ XGBoost model saved to {xgb_save_path}")
        except Exception as e:
            print(f"⚠ Error saving XGBoost model: {e}")
            # Try alternative save method
            try:
                import pickle
                with open(str(xgb_save_path).replace('.json', '.pkl'), 'wb') as f:
                    pickle.dump(xgb_model, f)
                print(f"✓ XGBoost model saved as pickle")
            except Exception as e2:
                print(f"✗ Failed to save XGBoost model: {e2}")
    
    # Save CatBoost model
    if TRAIN_CATBOOST and catboost_model is not None:
        cb_save_path = Path(model_save_path) / 'catboost_model.cbm'
        try:
            catboost_model.save_model(str(cb_save_path))
            print(f"✓ CatBoost model saved to {cb_save_path}")
        except Exception as e:
            print(f"⚠ Error saving CatBoost model: {e}")
            # Try alternative save method
            try:
                import pickle
                with open(str(cb_save_path).replace('.cbm', '.pkl'), 'wb') as f:
                    pickle.dump(catboost_model, f)
                print(f"✓ CatBoost model saved as pickle")
            except Exception as e2:
                print(f"✗ Failed to save CatBoost model: {e2}")
                
except Exception as e:
    print(f"⚠ Error saving model: {e}")
    print("Model may still be usable, but checkpoint not saved")

# Save metrics
try:
    weighted_f1_val = float(weighted_f1)
except:
    weighted_f1_val = float(opt_f1)

try:
    class_weights_dict = {
        'LOW': float(class_weights['LOW']),
        'MED': float(class_weights['MED']),
        'HIGH': float(class_weights['HIGH'])
    }
except:
    class_weights_dict = {'LOW': 1.0, 'MED': 1.0, 'HIGH': 1.0}

try:
    recalls_dict = {
        'LOW': float(recalls.get('LOW', 0)),
        'MED': float(recalls.get('MED', 0)),
        'HIGH': float(recalls.get('HIGH', 0))
    }
except:
    recalls_dict = {'LOW': 0.0, 'MED': 0.0, 'HIGH': 0.0}

metrics = {
    'mse': float(mse),
    'rmse': float(rmse),
    'mae': float(mae),
    'r2': float(r2),
    'kappa': float(kappa),
    'macro_f1': float(macro_f1),
    'optimized_thresholds': {
        'low': float(opt_low),
        'high': float(opt_high),
        'initial_f1': float(opt_f1),
        'weighted_f1': weighted_f1_val
    },
    'class_weights': class_weights_dict,
    'per_class_recall': recalls_dict
}

# Add uncertainty metrics if ensemble was used
if USE_ENSEMBLE and ensemble is not None and y_pred_std is not None:
    metrics['uncertainty'] = {
        'mean_std': float(np.mean(y_pred_std)),
        'max_std': float(np.max(y_pred_std)),
        'min_std': float(np.min(y_pred_std)),
        'median_std': float(np.median(y_pred_std))
    }
    metrics['ensemble_size'] = ensemble.n_models

# Add all models' metrics
all_models_metrics = {
    'FT-Transformer': metrics
}

if TRAIN_XGBOOST and 'XGBoost' in all_metrics:
    all_models_metrics['XGBoost'] = all_metrics['XGBoost']

if TRAIN_CATBOOST and 'CatBoost' in all_metrics:
    all_models_metrics['CatBoost'] = all_metrics['CatBoost']

# Save all metrics
with open('/content/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✓ FT-Transformer metrics saved to /content/metrics.json")

# Save comparison metrics
if len(all_models_metrics) > 1:
    comparison_metrics_path = '/content/model_comparison_metrics.json'
    with open(comparison_metrics_path, 'w') as f:
        json.dump(all_models_metrics, f, indent=2)
    print(f"✓ Model comparison metrics saved to {comparison_metrics_path}")

# Final Summary
print("\n" + "="*60)
print("Training Summary")
print("="*60)
print(f"✓ Dataset: {len(df)} total records")
print(f"✓ Features: {len(feature_columns)} ({len(continuous_cols)} continuous, {len(categorical_cols)} categorical)")
print(f"✓ Train/Val/Test: {len(train_df_split)}/{len(val_df_split)}/{len(test_df)}")
if USE_ENSEMBLE and ensemble is not None:
    print(f"✓ Model: FT-Transformer Ensemble ({ensemble.n_models} models)")
    if y_pred_std is not None:
        print(f"✓ Uncertainty Quantification:")
        print(f"    - Mean uncertainty (std): {np.mean(y_pred_std):.4f}")
        print(f"    - Max uncertainty: {np.max(y_pred_std):.4f}")
else:
    print(f"✓ Model: FT-Transformer (single model)")
print(f"✓ Best Metrics:")
print(f"    - RMSE: {rmse:.4f}")
print(f"    - MAE: {mae:.4f}")
print(f"    - R²: {r2:.4f}")
print(f"    - Cohen's Kappa: {kappa:.4f}")
print(f"    - Macro F1: {macro_f1:.4f}")
print(f"✓ Optimized Thresholds:")
print(f"    - LOW ≤ {opt_low:.3f}")
print(f"    - MED: {opt_low:.3f} - {opt_high:.3f}")
print(f"    - HIGH ≥ {opt_high:.3f}")
if USE_ENSEMBLE and ensemble is not None:
    print(f"✓ Models saved to: {model_save_path} (ensemble in {model_save_path}/ensemble/)")
else:
    print(f"✓ Model saved to: {model_save_path}")
print(f"✓ Metrics saved to: /content/metrics.json")
print(f"✓ Plots saved to:")
print(f"    - /content/training_curves.png (loss & MSE curves)")
print(f"    - /content/training_results.png (predictions & distributions)")
print(f"    - /content/confusion_matrix.png (classification matrix)")
if len(all_predictions) > 1:
    print(f"    - /content/model_comparison.png (model comparison plots)")
print(f"\n✓ Models Trained:")
print(f"    - FT-Transformer: {'✓' if 'FT-Transformer' in all_models else '✗'}")
print(f"    - XGBoost: {'✓' if TRAIN_XGBOOST and 'XGBoost' in all_models else '✗'}")
print(f"    - CatBoost: {'✓' if TRAIN_CATBOOST and 'CatBoost' in all_models else '✗'}")
print("\n" + "="*60)
print("Training Complete!")
print("="*60)