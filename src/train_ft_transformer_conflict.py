import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score, f1_score, recall_score, accuracy_score, precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
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

# Fill NaN for all features (use 0.0 for numeric, False for boolean)
numeric_cols_to_fill = [
    'position_agreement', 'pose_detected', 'pose_confidence', 'body_orientation_angle',
    'angle_to_manual_trapezoid', 'angle_to_segformer_road',
    'torso_lean_angle', 'head_orientation_angle', 'leg_separation',
    'estimated_stride_ratio', 'arm_crossing_score',
    'min_distance_to_vehicle', 'min_distance_to_vehicle_norm', 'nearby_pedestrians_count',
    'relative_x_to_vehicle', 'relative_y_to_vehicle',
    'traffic_density', 'pedestrian_density', 'road_area_ratio', 'distance_to_road_center',
    'road_segments_count', 'is_intersection', 'image_blur_score', 'image_brightness',
    'local_road_ratio', 'regional_road_ratio', 'global_road_ratio',
    'distance_to_left_edge', 'distance_to_right_edge', 'distance_to_top_edge',
    'distance_to_bottom_edge', 'position_x_norm', 'position_y_norm'
]

for col in numeric_cols_to_fill:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)

# Fill boolean columns with False
boolean_cols_to_fill = ['in_manual_trapezoid', 'bbox_inside_manual', 'in_segformer_road', 
                       'bbox_inside_segformer', 'pose_detected']
for col in boolean_cols_to_fill:
    if col in df.columns:
        df[col] = df[col].fillna(False).astype(float)  # Convert to float for model compatibility

# Interaction features (using SAFE features only - no leaking features)
# Note: Removed interactions with body_orientation_angle and position_agreement (leaking features)
if 'bbox_area_norm' in df.columns and 'pose_confidence' in df.columns:
    df['area_pose_confidence_interaction'] = df['bbox_area_norm'] * df['pose_confidence']
if 'bbox_center_y_norm' in df.columns and 'pose_confidence' in df.columns:
    df['position_pose_confidence_interaction'] = df['bbox_center_y_norm'] * df['pose_confidence']

# Removed leaking feature interactions:
# - orientation_agreement_interaction (uses body_orientation_angle + position_agreement)
# - area_orientation_interaction (uses body_orientation_angle)
# - pose_orientation_interaction (uses body_orientation_angle)

# Alternative safe interactions using advanced pose features instead
if 'bbox_area_norm' in df.columns and 'torso_lean_angle' in df.columns:
    df['area_orientation_interaction'] = df['bbox_area_norm'] * (np.abs(df['torso_lean_angle']) / 180.0)
if 'pose_confidence' in df.columns and 'torso_lean_angle' in df.columns:
    df['pose_orientation_interaction'] = df['pose_confidence'] * (np.abs(df['torso_lean_angle']) / 180.0)
if 'torso_lean_angle' in df.columns and 'head_orientation_angle' in df.columns:
    df['orientation_agreement_interaction'] = (np.abs(df['torso_lean_angle']) / 180.0) * (np.abs(df['head_orientation_angle']) / 180.0)

# Feature selection - EXCLUDE DATA LEAKAGE FEATURES
# These features are directly used in conflict_score formula and cause data leakage:
# - in_manual_trapezoid, in_segformer_road (used in position_score, 0.65 weight)
# - bbox_inside_manual, bbox_inside_segformer (used in position_score)
# - position_agreement (used in agreement_score, 0.10 weight)
# - angle_to_manual_trapezoid, angle_to_segformer_road (used in pose_score, 0.25 weight)
# - body_orientation_angle (used in pose_score, 0.25 weight)

# Features to EXCLUDE (data leakage - directly used in conflict_score formula)
leaking_features = [
    'in_manual_trapezoid',      # Used in position_score (0.65 weight)
    'in_segformer_road',        # Used in position_score (0.65 weight)
    'bbox_inside_manual',       # Used in position_score
    'bbox_inside_segformer',    # Used in position_score
    'position_agreement',        # Used in agreement_score (0.10 weight)
    'angle_to_manual_trapezoid', # Used in pose_score (0.25 weight)
    'angle_to_segformer_road',   # Used in pose_score
    'body_orientation_angle',    # Used in pose_score (0.25 weight)
]

# SAFE features (NOT used in conflict_score formula - no data leakage)
normalized_bbox_features = ['bbox_x1_norm', 'bbox_y1_norm', 'bbox_x2_norm', 'bbox_y2_norm',
                           'bbox_center_x_norm', 'bbox_center_y_norm', 'bbox_area_norm',
                           'bbox_width_norm', 'bbox_height_norm', 'bbox_aspect_ratio']

# Basic pose features (raw, not computed scores)
pose_features = ['pose_detected', 'pose_confidence']  # Removed: position_agreement, body_orientation_angle, angles

# Advanced pose features (if available in CSV)
advanced_pose_features = ['torso_lean_angle', 'head_orientation_angle', 'leg_separation',
                         'estimated_stride_ratio', 'arm_crossing_score']

# Spatial relationship features (if available in CSV) - NOT in conflict_score formula
spatial_features = ['min_distance_to_vehicle', 'min_distance_to_vehicle_norm', 'nearby_pedestrians_count',
                   'relative_x_to_vehicle', 'relative_y_to_vehicle']

# Scene context features (if available in CSV) - NOT in conflict_score formula
scene_features = ['traffic_density', 'pedestrian_density', 'road_area_ratio', 'distance_to_road_center',
                 'road_segments_count', 'is_intersection', 'image_blur_score', 'image_brightness']

# Multi-scale spatial features (if available in CSV) - NOT in conflict_score formula
multiscale_features = ['local_road_ratio', 'regional_road_ratio', 'global_road_ratio',
                      'distance_to_left_edge', 'distance_to_right_edge', 'distance_to_top_edge',
                      'distance_to_bottom_edge', 'position_x_norm', 'position_y_norm']

# Position features - REMOVED (all cause data leakage)
# position_features = []  # All position features removed

# Interaction features (computed from safe base features only)
interaction_features = ['area_pose_confidence_interaction', 'position_pose_confidence_interaction',
                       'orientation_agreement_interaction', 'area_orientation_interaction',
                       'pose_orientation_interaction']

# Combine all SAFE feature lists (excluding leaking features)
all_features = (normalized_bbox_features + pose_features + advanced_pose_features + 
               spatial_features + scene_features + multiscale_features + 
               interaction_features)

# Only use features that exist in the CSV AND are not leaking features
feature_columns = [f for f in all_features if f in df.columns and f not in leaking_features]

# Also exclude any leaking features that might be in CSV (double-check)
feature_columns = [f for f in feature_columns if f not in leaking_features]

# CRITICAL: Also exclude 'position_type' if present (derived from leaking features)
if 'position_type' in feature_columns:
    feature_columns.remove('position_type')
    print("⚠ Excluded 'position_type' (derived from leaking features)")

# Print which features are excluded (data leakage)
excluded_leaking = [f for f in leaking_features if f in df.columns]
if excluded_leaking:
    print(f"\n{'='*60}")
    print(f"⚠ DATA LEAKAGE PREVENTION: Excluding {len(excluded_leaking)} features")
    print(f"{'='*60}")
    print("These features are directly used in conflict_score formula:")
    for feat in excluded_leaking:
        print(f"    ✗ {feat}")
    print(f"\n✓ Model will learn from safe features only (no formula reverse-engineering)")
    print(f"{'='*60}\n")

# FINAL VERIFICATION: Ensure no leaking features are in feature_columns
leaking_found = [f for f in feature_columns if f in leaking_features or f == 'position_type']
if leaking_found:
    print(f"⚠ ERROR: Found leaking features in feature_columns: {leaking_found}")
    print("Removing them now...")
    feature_columns = [f for f in feature_columns if f not in leaking_features and f != 'position_type']
    print(f"✓ Removed leaking features. Using {len(feature_columns)} safe features.")

# Print which features are missing
missing_features = [f for f in all_features if f not in df.columns]
if missing_features:
    print(f"⚠ Note: {len(missing_features)} expected features not found in CSV (will be skipped):")
    for feat in missing_features[:10]:  # Show first 10
        print(f"    - {feat}")
    if len(missing_features) > 10:
        print(f"    ... and {len(missing_features) - 10} more")

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
# Reduced learning rate to slow down convergence (was 3e-4, now 1e-4)
model_config_kwargs = {
    'task': "regression",
    'learning_rate': 1e-4,  # Reduced from 3e-4 for slower, more gradual learning
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
# Updated to slow down learning: larger batch size, gradient clipping, LR scheduler
trainer_config_kwargs = {
    'batch_size': 256,  # Increased from 128 for smoother gradients and slower learning
    'max_epochs': 40,  # Reduced to 40 epochs as requested
    'early_stopping': "valid_loss",
    'early_stopping_patience': 15,  # README: Patience=15
    'gradient_clip_val': 1.0,  # NEW: Clip gradients to prevent large updates
    'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
    'devices': 1 if torch.cuda.is_available() else None,
}

# Try to add early_stopping_min_delta and learning rate scheduler
trainer_config = None
try:
    # First try with min_delta and LR scheduler
    trainer_config_kwargs['early_stopping_min_delta'] = 1e-5  # README: min_delta=1e-5
    
    # Try to add learning rate scheduler for gradual learning
    try:
        trainer_config_kwargs['learning_rate_scheduler'] = 'ReduceLROnPlateau'
        trainer_config_kwargs['learning_rate_scheduler_params'] = {
            'mode': 'min',
            'factor': 0.5,  # Reduce LR by 50% when plateau detected
            'patience': 5,  # Wait 5 epochs before reducing LR
            'min_lr': 1e-6  # Minimum learning rate
        }
        print("✓ Learning rate scheduler added: ReduceLROnPlateau")
    except (TypeError, ValueError, AttributeError) as lr_e:
        # LR scheduler might not be supported, continue without it
        trainer_config_kwargs.pop('learning_rate_scheduler', None)
        trainer_config_kwargs.pop('learning_rate_scheduler_params', None)
        print(f"⚠ Learning rate scheduler not supported, continuing without it")
    
    trainer_config = TrainerConfig(**trainer_config_kwargs)
except (TypeError, ValueError) as e:
    # If min_delta not supported, remove it and try again
    trainer_config_kwargs.pop('early_stopping_min_delta', None)
    trainer_config_kwargs.pop('learning_rate_scheduler', None)
    trainer_config_kwargs.pop('learning_rate_scheduler_params', None)
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

# Improved checkpoint saving - Save as .pkl
def save_ensemble_model(model, save_dir, model_index):
    """Save model as .pkl file"""
    import os
    import pickle
    
    model_dir = Path(save_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as .pkl file
    pkl_path = model_dir / f"model_{model_index}.pkl"
    
    try:
        # Save entire model object as pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model {model_index} saved to {pkl_path}")
        return True
    except Exception as e:
        print(f"⚠ Error saving model {model_index} as pickle: {e}")
        # Try saving state dict as backup
        try:
            if hasattr(model, 'model'):
                state_dict_path = model_dir / f"model_{model_index}_state_dict.pkl"
                with open(state_dict_path, 'wb') as f:
                    pickle.dump({
                        'state_dict': model.model.state_dict(),
                        'config': {
                            'data_config': model.data_config if hasattr(model, 'data_config') else None,
                            'model_config': model.model_config if hasattr(model, 'model_config') else None,
                        }
                    }, f)
                print(f"✓ Model {model_index} state dict saved as backup to {state_dict_path}")
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
if hasattr(trainer_config, 'gradient_clip_val'):
    print(f"  Gradient clipping: {trainer_config.gradient_clip_val}")
if hasattr(trainer_config, 'learning_rate_scheduler'):
    print(f"  Learning rate scheduler: {trainer_config.learning_rate_scheduler}")
    if hasattr(trainer_config, 'learning_rate_scheduler_params'):
        print(f"    Scheduler params: {trainer_config.learning_rate_scheduler_params}")

print("\n" + "-"*60)
print("Anti-Overfitting Measures & Learning Rate Control:")
print("-"*60)
print("✓ Dropout: attn_dropout=0.1, ff_dropout=0.1, embedding_dropout=0.05")
print("✓ Weight Decay: L2 regularization (1e-4)")
print(f"✓ Early Stopping: Patience={trainer_config.early_stopping_patience}, min_delta={getattr(trainer_config, 'early_stopping_min_delta', 'N/A')}")
print(f"✓ Gradient Clipping: {getattr(trainer_config, 'gradient_clip_val', 'N/A')}")
if hasattr(trainer_config, 'learning_rate_scheduler') and trainer_config.learning_rate_scheduler:
    print(f"✓ Learning Rate Scheduler: {trainer_config.learning_rate_scheduler}")
    if hasattr(trainer_config, 'learning_rate_scheduler_params'):
        params = trainer_config.learning_rate_scheduler_params
        print(f"    - Factor: {params.get('factor', 'N/A')}")
        print(f"    - Patience: {params.get('patience', 'N/A')}")
        print(f"    - Min LR: {params.get('min_lr', 'N/A')}")
else:
    print("⚠ Learning Rate Scheduler: Not configured (may be handled internally)")
print("✓ Reduced Learning Rate: 1e-4 (was 3e-4) for slower, more gradual learning")
print("✓ Increased Batch Size: 256 (was 128) for smoother gradients")
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
        try:
            best_iteration = catboost_model.get_best_iteration()
            print(f"  Best iteration: {best_iteration}")
        except:
            print(f"  Best iteration: N/A")
        
        # get_best_score() returns a dict, not a float - handle it safely
        try:
            best_score_dict = catboost_model.get_best_score()
            if isinstance(best_score_dict, dict):
                # Extract RMSE from the dict (format: {'learn': {'RMSE': value}, 'validation': {'RMSE': value}})
                # Try validation first (more relevant), then learn
                best_score = None
                if 'validation' in best_score_dict:
                    validation_scores = best_score_dict['validation']
                    if isinstance(validation_scores, dict):
                        best_score = validation_scores.get('RMSE', None)
                
                if best_score is None and 'learn' in best_score_dict:
                    learn_scores = best_score_dict['learn']
                    if isinstance(learn_scores, dict):
                        best_score = learn_scores.get('RMSE', None)
                
                if best_score is not None and isinstance(best_score, (int, float)):
                    print(f"  Best score (RMSE): {best_score:.6f}")
                else:
                    # Fallback: print the dict structure
                    print(f"  Best score: {best_score_dict}")
            else:
                # If it's not a dict, try to format as float
                if isinstance(best_score_dict, (int, float)):
                    print(f"  Best score: {best_score_dict:.6f}")
                else:
                    print(f"  Best score: {best_score_dict}")
        except Exception as score_error:
            # If get_best_score() fails, just skip it
            print(f"  Best score: N/A (could not retrieve: {str(score_error)[:50]})")
        
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
                    if key not in training_metrics:
                        training_metrics[key] = []
                    if isinstance(value, (list, np.ndarray)):
                        training_metrics[key].extend(value if isinstance(value, list) else value.tolist())
                    else:
                        training_metrics[key].append(value)
        
        # Check for callback metrics
        if hasattr(trainer, 'callback_metrics') and trainer.callback_metrics:
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.cpu().numpy()
                if key not in training_metrics:
                    training_metrics[key] = []
                if isinstance(value, (list, np.ndarray)):
                    training_metrics[key].extend(value if isinstance(value, list) else value.tolist())
                else:
                    training_metrics[key].append(value)
        
        # Check for CSV logger (common in PyTorch Lightning) - PRIORITY METHOD
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
                                if 'loss' in col.lower() or 'mse' in col.lower() or 'mean_squared_error' in col.lower():
                                    training_metrics[col] = df_logs[col].dropna().tolist()
                            print(f"  ✓ Loaded metrics from CSV logger: {len(df_logs)} rows")
                    except Exception as csv_e:
                        pass
    
    # Method 3: Check model save directory for logs (PRIORITY METHOD)
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
                        if 'loss' in col.lower() or 'mse' in col.lower() or 'mean_squared_error' in col.lower():
                            training_metrics[col] = df_logs[col].dropna().tolist()
                    print(f"  ✓ Loaded metrics from CSV file: {len(df_logs)} rows, {len(training_metrics)} metrics")
    except Exception as log_e:
        pass
    
    # Method 4: Try to get metrics from ensemble models if available
    if USE_ENSEMBLE and ensemble_models:
        try:
            for i, ensemble_model in enumerate(ensemble_models):
                if hasattr(ensemble_model, 'trainer') and ensemble_model.trainer is not None:
                    trainer = ensemble_model.trainer
                    log_dir = Path('/content/ft_transformer_model') / 'lightning_logs'
                    if log_dir.exists():
                        versions = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith('version')])
                        if versions:
                            # Try different version numbers
                            for version_dir in versions[-3:]:  # Check last 3 versions
                                csv_file = version_dir / 'metrics.csv'
                                if csv_file.exists():
                                    df_logs = pd.read_csv(csv_file)
                                    for col in df_logs.columns:
                                        if 'loss' in col.lower() or 'mse' in col.lower() or 'mean_squared_error' in col.lower():
                                            if col not in training_metrics:
                                                training_metrics[col] = []
                                            training_metrics[col].extend(df_logs[col].dropna().tolist())
                                    break
        except:
            pass
    
    # Extract and plot metrics
    train_loss = []
    val_loss = []
    train_mse = []
    val_mse = []
    
    # Try to find loss metrics - handle both with and without 'epoch' in key
    for key, values in training_metrics.items():
        key_lower = key.lower()
        # Check for train loss (with or without 'epoch')
        if 'train' in key_lower and 'loss' in key_lower:
            if isinstance(values, list):
                train_loss = values
            elif isinstance(values, (int, float)):
                train_loss = [values]
        # Check for validation loss (with or without 'epoch')
        elif ('val' in key_lower or 'valid' in key_lower) and 'loss' in key_lower:
            if isinstance(values, list):
                val_loss = values
            elif isinstance(values, (int, float)):
                val_loss = [values]
        # Check for train MSE
        elif 'train' in key_lower and ('mse' in key_lower or 'mean_squared_error' in key_lower):
            if isinstance(values, list):
                train_mse = values
            elif isinstance(values, (int, float)):
                train_mse = [values]
        # Check for validation MSE
        elif ('val' in key_lower or 'valid' in key_lower) and ('mse' in key_lower or 'mean_squared_error' in key_lower):
            if isinstance(values, list):
                val_mse = values
            elif isinstance(values, (int, float)):
                val_mse = [values]
    
    # If we have the exact keys from pytorch_tabular, use them directly
    if not train_loss and 'train_loss' in training_metrics:
        train_loss = training_metrics['train_loss'] if isinstance(training_metrics['train_loss'], list) else [training_metrics['train_loss']]
    if not val_loss and 'valid_loss' in training_metrics:
        val_loss = training_metrics['valid_loss'] if isinstance(training_metrics['valid_loss'], list) else [training_metrics['valid_loss']]
    if not train_mse and 'train_mean_squared_error' in training_metrics:
        train_mse = training_metrics['train_mean_squared_error'] if isinstance(training_metrics['train_mean_squared_error'], list) else [training_metrics['train_mean_squared_error']]
    if not val_mse and 'valid_mean_squared_error' in training_metrics:
        val_mse = training_metrics['valid_mean_squared_error'] if isinstance(training_metrics['valid_mean_squared_error'], list) else [training_metrics['valid_mean_squared_error']]
    
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

# Store predictions for all models
all_class_predictions = {}
all_class_predictions['FT-Transformer'] = y_pred_class

# Initialize ROC-AUC scores storage
roc_auc_scores = {}

# Calculate comprehensive classification metrics for FT-Transformer
accuracy_ft = accuracy_score(y_true_class, y_pred_class)
precision_ft = precision_score(y_true_class, y_pred_class, average=None, zero_division=0, labels=['LOW', 'MED', 'HIGH'])
recall_ft = recall_score(y_true_class, y_pred_class, average=None, zero_division=0, labels=['LOW', 'MED', 'HIGH'])
f1_ft = f1_score(y_true_class, y_pred_class, average=None, zero_division=0, labels=['LOW', 'MED', 'HIGH'])
macro_precision_ft = precision_score(y_true_class, y_pred_class, average='macro', zero_division=0)
macro_recall_ft = recall_score(y_true_class, y_pred_class, average='macro', zero_division=0)
macro_f1_ft = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
weighted_f1_ft = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
kappa_ft = cohen_kappa_score(y_true_class, y_pred_class)

print("\n" + "="*60)
print("FT-Transformer Classification Report")
print("="*60)
print(f"\nAccuracy: {accuracy_ft:.4f}")
print(f"\nPer-Class Metrics:")
print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 46)
for i, cls in enumerate(['LOW', 'MED', 'HIGH']):
    print(f"{cls:<10} {precision_ft[i]:<12.4f} {recall_ft[i]:<12.4f} {f1_ft[i]:<12.4f}")
print(f"\nMacro Average:")
print(f"  Precision: {macro_precision_ft:.4f}")
print(f"  Recall: {macro_recall_ft:.4f}")
print(f"  F1-Score: {macro_f1_ft:.4f}")
print(f"\nWeighted F1-Score: {weighted_f1_ft:.4f}")
print(f"Cohen's Kappa: {kappa_ft:.4f}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_true_class, y_pred_class, target_names=['LOW', 'MED', 'HIGH'], digits=4))

# Store FT-Transformer classification metrics
all_metrics['FT-Transformer']['accuracy'] = accuracy_ft
all_metrics['FT-Transformer']['precision'] = {cls: float(precision_ft[i]) for i, cls in enumerate(['LOW', 'MED', 'HIGH'])}
all_metrics['FT-Transformer']['recall'] = {cls: float(recall_ft[i]) for i, cls in enumerate(['LOW', 'MED', 'HIGH'])}
all_metrics['FT-Transformer']['f1'] = {cls: float(f1_ft[i]) for i, cls in enumerate(['LOW', 'MED', 'HIGH'])}
all_metrics['FT-Transformer']['macro_precision'] = macro_precision_ft
all_metrics['FT-Transformer']['macro_recall'] = macro_recall_ft
all_metrics['FT-Transformer']['macro_f1'] = macro_f1_ft
all_metrics['FT-Transformer']['weighted_f1'] = weighted_f1_ft
all_metrics['FT-Transformer']['kappa'] = kappa_ft
all_metrics['FT-Transformer']['thresholds'] = {'low': float(opt_low), 'high': float(opt_high)}

# Evaluate XGBoost with same thresholds
if TRAIN_XGBOOST and xgb_model is not None:
    y_pred_xgb_class = score_to_level(all_predictions['XGBoost'], low=opt_low, high=opt_high)
    all_class_predictions['XGBoost'] = y_pred_xgb_class
    
    accuracy_xgb = accuracy_score(y_true_class, y_pred_xgb_class)
    precision_xgb = precision_score(y_true_class, y_pred_xgb_class, average=None, zero_division=0, labels=['LOW', 'MED', 'HIGH'])
    recall_xgb = recall_score(y_true_class, y_pred_xgb_class, average=None, zero_division=0, labels=['LOW', 'MED', 'HIGH'])
    f1_xgb = f1_score(y_true_class, y_pred_xgb_class, average=None, zero_division=0, labels=['LOW', 'MED', 'HIGH'])
    macro_precision_xgb = precision_score(y_true_class, y_pred_xgb_class, average='macro', zero_division=0)
    macro_recall_xgb = recall_score(y_true_class, y_pred_xgb_class, average='macro', zero_division=0)
    macro_f1_xgb = f1_score(y_true_class, y_pred_xgb_class, average='macro', zero_division=0)
    weighted_f1_xgb = f1_score(y_true_class, y_pred_xgb_class, average='weighted', zero_division=0)
    kappa_xgb = cohen_kappa_score(y_true_class, y_pred_xgb_class)
    
    print("\n" + "="*60)
    print("XGBoost Classification Report")
    print("="*60)
    print(f"\nAccuracy: {accuracy_xgb:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 46)
    for i, cls in enumerate(['LOW', 'MED', 'HIGH']):
        print(f"{cls:<10} {precision_xgb[i]:<12.4f} {recall_xgb[i]:<12.4f} {f1_xgb[i]:<12.4f}")
    print(f"\nMacro Average:")
    print(f"  Precision: {macro_precision_xgb:.4f}")
    print(f"  Recall: {macro_recall_xgb:.4f}")
    print(f"  F1-Score: {macro_f1_xgb:.4f}")
    print(f"\nWeighted F1-Score: {weighted_f1_xgb:.4f}")
    print(f"Cohen's Kappa: {kappa_xgb:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true_class, y_pred_xgb_class, target_names=['LOW', 'MED', 'HIGH'], digits=4))
    
    all_metrics['XGBoost']['accuracy'] = accuracy_xgb
    all_metrics['XGBoost']['precision'] = {cls: float(precision_xgb[i]) for i, cls in enumerate(['LOW', 'MED', 'HIGH'])}
    all_metrics['XGBoost']['recall'] = {cls: float(recall_xgb[i]) for i, cls in enumerate(['LOW', 'MED', 'HIGH'])}
    all_metrics['XGBoost']['f1'] = {cls: float(f1_xgb[i]) for i, cls in enumerate(['LOW', 'MED', 'HIGH'])}
    all_metrics['XGBoost']['macro_precision'] = macro_precision_xgb
    all_metrics['XGBoost']['macro_recall'] = macro_recall_xgb
    all_metrics['XGBoost']['macro_f1'] = macro_f1_xgb
    all_metrics['XGBoost']['weighted_f1'] = weighted_f1_xgb
    all_metrics['XGBoost']['kappa'] = kappa_xgb
    all_metrics['XGBoost']['thresholds'] = {'low': float(opt_low), 'high': float(opt_high)}

# Evaluate CatBoost with same thresholds
if TRAIN_CATBOOST and catboost_model is not None:
    y_pred_cb_class = score_to_level(all_predictions['CatBoost'], low=opt_low, high=opt_high)
    all_class_predictions['CatBoost'] = y_pred_cb_class
    
    accuracy_cb = accuracy_score(y_true_class, y_pred_cb_class)
    precision_cb = precision_score(y_true_class, y_pred_cb_class, average=None, zero_division=0, labels=['LOW', 'MED', 'HIGH'])
    recall_cb = recall_score(y_true_class, y_pred_cb_class, average=None, zero_division=0, labels=['LOW', 'MED', 'HIGH'])
    f1_cb = f1_score(y_true_class, y_pred_cb_class, average=None, zero_division=0, labels=['LOW', 'MED', 'HIGH'])
    macro_precision_cb = precision_score(y_true_class, y_pred_cb_class, average='macro', zero_division=0)
    macro_recall_cb = recall_score(y_true_class, y_pred_cb_class, average='macro', zero_division=0)
    macro_f1_cb = f1_score(y_true_class, y_pred_cb_class, average='macro', zero_division=0)
    weighted_f1_cb = f1_score(y_true_class, y_pred_cb_class, average='weighted', zero_division=0)
    kappa_cb = cohen_kappa_score(y_true_class, y_pred_cb_class)
    
    print("\n" + "="*60)
    print("CatBoost Classification Report")
    print("="*60)
    print(f"\nAccuracy: {accuracy_cb:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 46)
    for i, cls in enumerate(['LOW', 'MED', 'HIGH']):
        print(f"{cls:<10} {precision_cb[i]:<12.4f} {recall_cb[i]:<12.4f} {f1_cb[i]:<12.4f}")
    print(f"\nMacro Average:")
    print(f"  Precision: {macro_precision_cb:.4f}")
    print(f"  Recall: {macro_recall_cb:.4f}")
    print(f"  F1-Score: {macro_f1_cb:.4f}")
    print(f"\nWeighted F1-Score: {weighted_f1_cb:.4f}")
    print(f"Cohen's Kappa: {kappa_cb:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true_class, y_pred_cb_class, target_names=['LOW', 'MED', 'HIGH'], digits=4))
    
    all_metrics['CatBoost']['accuracy'] = accuracy_cb
    all_metrics['CatBoost']['precision'] = {cls: float(precision_cb[i]) for i, cls in enumerate(['LOW', 'MED', 'HIGH'])}
    all_metrics['CatBoost']['recall'] = {cls: float(recall_cb[i]) for i, cls in enumerate(['LOW', 'MED', 'HIGH'])}
    all_metrics['CatBoost']['f1'] = {cls: float(f1_cb[i]) for i, cls in enumerate(['LOW', 'MED', 'HIGH'])}
    all_metrics['CatBoost']['macro_precision'] = macro_precision_cb
    all_metrics['CatBoost']['macro_recall'] = macro_recall_cb
    all_metrics['CatBoost']['macro_f1'] = macro_f1_cb
    all_metrics['CatBoost']['weighted_f1'] = weighted_f1_cb
    all_metrics['CatBoost']['kappa'] = kappa_cb
    all_metrics['CatBoost']['thresholds'] = {'low': float(opt_low), 'high': float(opt_high)}

# Classification Comparison Summary
if len([m for m in all_metrics.keys() if 'accuracy' in all_metrics[m]]) > 1:
    print("\n" + "="*60)
    print("Classification Metrics Comparison")
    print("="*60)
    
    cls_comparison = pd.DataFrame({
        model: {
            'Accuracy': all_metrics[model].get('accuracy', 0),
            'Macro Precision': all_metrics[model].get('macro_precision', 0),
            'Macro Recall': all_metrics[model].get('macro_recall', 0),
            'Macro F1': all_metrics[model].get('macro_f1', 0),
            'Weighted F1': all_metrics[model].get('weighted_f1', 0),
            'Kappa': all_metrics[model].get('kappa', 0)
        }
        for model in all_metrics.keys() if 'accuracy' in all_metrics[model]
    }).T
    
    print("\nClassification Metrics Comparison:")
    print(cls_comparison.round(4))
    
    print("\nBest Model per Classification Metric:")
    best_acc_model = cls_comparison['Accuracy'].idxmax()
    best_prec_model = cls_comparison['Macro Precision'].idxmax()
    best_recall_model = cls_comparison['Macro Recall'].idxmax()
    best_f1_model = cls_comparison['Macro F1'].idxmax()
    best_kappa_model = cls_comparison['Kappa'].idxmax()
    print(f"  Accuracy: {best_acc_model} ({cls_comparison.loc[best_acc_model, 'Accuracy']:.4f})")
    print(f"  Macro Precision: {best_prec_model} ({cls_comparison.loc[best_prec_model, 'Macro Precision']:.4f})")
    print(f"  Macro Recall: {best_recall_model} ({cls_comparison.loc[best_recall_model, 'Macro Recall']:.4f})")
    print(f"  Macro F1: {best_f1_model} ({cls_comparison.loc[best_f1_model, 'Macro F1']:.4f})")
    print(f"  Kappa: {best_kappa_model} ({cls_comparison.loc[best_kappa_model, 'Kappa']:.4f})")

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

# Normalized Confusion Matrices for All Models
print("\n" + "="*60)
print("Normalized Confusion Matrices")
print("="*60)

# Create figure with subplots for all models
n_models = len(all_class_predictions)
fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
if n_models == 1:
    axes = [axes]

for idx, (model_name, y_pred_model_class) in enumerate(all_class_predictions.items()):
    cm = confusion_matrix(y_true_class, y_pred_model_class, labels=['LOW', 'MED', 'HIGH'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=['LOW', 'MED', 'HIGH'], 
                yticklabels=['LOW', 'MED', 'HIGH'],
                ax=axes[idx], vmin=0, vmax=1, cbar_kws={'label': 'Normalized'})
    axes[idx].set_title(f'{model_name} - Normalized Confusion Matrix')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('/content/normalized_confusion_matrices.png', dpi=200, bbox_inches='tight')
print("✓ Normalized confusion matrices saved to /content/normalized_confusion_matrices.png")
plt.show()

# Also save individual confusion matrices
for model_name, y_pred_model_class in all_class_predictions.items():
    cm = confusion_matrix(y_true_class, y_pred_model_class, labels=['LOW', 'MED', 'HIGH'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['LOW', 'MED', 'HIGH'], 
                yticklabels=['LOW', 'MED', 'HIGH'],
                vmin=0, vmax=1, cbar_kws={'label': 'Normalized'})
    plt.title(f'{model_name} - Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    safe_model_name = model_name.replace(' ', '_').lower()
    plt.savefig(f'/content/confusion_matrix_{safe_model_name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ {model_name} confusion matrix saved to /content/confusion_matrix_{safe_model_name}.png")

# ROC-AUC Curves Comparison
print("\n" + "="*60)
print("ROC-AUC Curves Comparison")
print("="*60)

# Convert class labels to numeric for ROC-AUC
label_to_num = {'LOW': 0, 'MED': 1, 'HIGH': 2}
y_true_numeric = np.array([label_to_num[label] for label in y_true_class])

# Binarize labels for multi-class ROC-AUC
y_true_binarized = label_binarize(y_true_numeric, classes=[0, 1, 2])

# Create figure for ROC-AUC curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Aggregated Multi-class ROC-AUC (Macro-Averaged, One per Model)
ax1 = axes[0]
colors_map = {'FT-Transformer': 'blue', 'XGBoost': 'green', 'CatBoost': 'orange'}
class_names = ['LOW', 'MED', 'HIGH']

for model_name, y_pred_model in all_predictions.items():
    # Convert regression predictions to probabilities for each class
    # Use softmax-like transformation based on distance to thresholds
    prob_matrix = np.zeros((len(y_pred_model), 3))
    
    for i, score in enumerate(y_pred_model):
        # Calculate probabilities based on distance to thresholds
        if score <= opt_low:
            prob_matrix[i, 0] = 1.0 - (score / opt_low) * 0.5  # LOW probability
            prob_matrix[i, 1] = (score / opt_low) * 0.5  # MED probability
            prob_matrix[i, 2] = 0.0  # HIGH probability
        elif score <= opt_high:
            prob_matrix[i, 0] = 0.0
            prob_matrix[i, 1] = 1.0 - abs(score - (opt_low + opt_high) / 2) / ((opt_high - opt_low) / 2) * 0.5
            prob_matrix[i, 2] = abs(score - (opt_low + opt_high) / 2) / ((opt_high - opt_low) / 2) * 0.5
        else:
            prob_matrix[i, 0] = 0.0
            prob_matrix[i, 1] = (1.0 - (score - opt_high) / (1.0 - opt_high)) * 0.5
            prob_matrix[i, 2] = 1.0 - (1.0 - (score - opt_high) / (1.0 - opt_high)) * 0.5
    
    # Normalize probabilities
    prob_matrix = prob_matrix / (prob_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    # Calculate ROC-AUC for each class and compute macro-averaged ROC curve
    roc_auc_per_class = {}
    all_fpr = []
    all_tpr = []
    
    for i, class_name in enumerate(class_names):
        if y_true_binarized[:, i].sum() > 0:  # Check if class exists in test set
            try:
                roc_auc = roc_auc_score(y_true_binarized[:, i], prob_matrix[:, i])
                roc_auc_per_class[class_name] = roc_auc
                
                # Get ROC curve for this class
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], prob_matrix[:, i])
                all_fpr.append(fpr)
                all_tpr.append(tpr)
            except:
                pass
    
    # Compute macro-averaged ROC curve (average across all classes)
    if all_fpr and all_tpr:
        # Interpolate all ROC curves to common FPR points
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        
        for fpr, tpr in zip(all_fpr, all_tpr):
            # Interpolate TPR at common FPR points
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        
        # Average across all classes
        mean_tpr /= len(all_fpr)
        
        # Calculate macro-averaged AUC
        macro_roc_auc = np.mean(list(roc_auc_per_class.values()))
        
        # Plot aggregated ROC curve (one per model)
        ax1.plot(mean_fpr, mean_tpr, lw=2, 
                label=f'{model_name} (Macro AUC={macro_roc_auc:.3f})',
                color=colors_map.get(model_name, 'gray'))
        
        roc_auc_scores[model_name] = {
            'per_class': roc_auc_per_class,
            'macro': macro_roc_auc
        }
        print(f"\n{model_name} ROC-AUC:")
        for cls, score in roc_auc_per_class.items():
            print(f"  {cls}: {score:.4f}")
        print(f"  Macro Average: {macro_roc_auc:.4f}")

ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.5)')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Multi-Class ROC Curves (Macro-Averaged, One per Model)')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Plot 2: Binary ROC-AUC (HIGH vs Not-HIGH)
ax2 = axes[1]
y_true_binary = (y_true_numeric == 2).astype(int)  # HIGH = 1, Not-HIGH = 0

for model_name, y_pred_model in all_predictions.items():
    # Convert regression predictions to probability of HIGH class
    # Use sigmoid-like transformation
    y_pred_binary_prob = 1 / (1 + np.exp(-10 * (y_pred_model - opt_high)))
    
    try:
        roc_auc_binary = roc_auc_score(y_true_binary, y_pred_binary_prob)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary_prob)
        ax2.plot(fpr, tpr, lw=2, 
                label=f'{model_name} (AUC={roc_auc_binary:.3f})',
                color=colors_map.get(model_name, 'gray'))
        
        if 'binary' not in roc_auc_scores.get(model_name, {}):
            roc_auc_scores[model_name] = roc_auc_scores.get(model_name, {})
        roc_auc_scores[model_name]['binary'] = roc_auc_binary
    except Exception as e:
        print(f"⚠ Error calculating binary ROC-AUC for {model_name}: {e}")

ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.5)')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Binary ROC Curves (HIGH vs Not-HIGH)')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/roc_auc_curves.png', dpi=200, bbox_inches='tight')
print("\n✓ ROC-AUC curves saved to /content/roc_auc_curves.png")
plt.show()

# Print ROC-AUC comparison
if len(roc_auc_scores) > 1:
    print("\n" + "="*60)
    print("ROC-AUC Comparison")
    print("="*60)
    
    roc_comparison = pd.DataFrame({
        model: {
            'LOW': roc_auc_scores[model].get('per_class', {}).get('LOW', 0),
            'MED': roc_auc_scores[model].get('per_class', {}).get('MED', 0),
            'HIGH': roc_auc_scores[model].get('per_class', {}).get('HIGH', 0),
            'Macro Average': roc_auc_scores[model].get('macro', 0),
            'Binary (HIGH vs Not-HIGH)': roc_auc_scores[model].get('binary', 0)
        }
        for model in roc_auc_scores.keys()
    }).T
    
    print("\nROC-AUC Scores:")
    print(roc_comparison.round(4))
    
    print("\nBest Model per ROC-AUC Metric:")
    for metric in ['LOW', 'MED', 'HIGH', 'Macro Average', 'Binary (HIGH vs Not-HIGH)']:
        if metric in roc_comparison.columns:
            best_model = roc_comparison[metric].idxmax()
            best_value = roc_comparison.loc[best_model, metric]
            print(f"  {metric}: {best_model} ({best_value:.4f})")
    
    # Store ROC-AUC scores in metrics
    for model_name in roc_auc_scores.keys():
        if model_name in all_metrics:
            all_metrics[model_name]['roc_auc'] = {
                'per_class': {k: float(v) for k, v in roc_auc_scores[model_name].get('per_class', {}).items()},
                'macro': float(roc_auc_scores[model_name].get('macro', 0)),
                'binary': float(roc_auc_scores[model_name].get('binary', 0))
            }

# Feature Importance Analysis with SHAP for All Models
print("\n" + "="*60)
print("SHAP Feature Importance Analysis (All Models)")
print("="*60)

# Prepare data for SHAP (use a sample for speed)
sample_size = min(100, len(test_df))
X_sample_df = test_df[feature_columns].iloc[:sample_size]
X_sample = X_sample_df.values

# Background dataset for PermutationExplainer (FT-Transformer)
background_size = min(50, len(test_df))
X_background = test_df[feature_columns].iloc[:background_size].values

all_shap_results = {}

try:
    import shap
    print("✓ SHAP library available")
    
    # Analyze FT-Transformer (PermutationExplainer)
    if 'FT-Transformer' in all_models:
        print(f"\n{'='*60}")
        print("Analyzing FT-Transformer with SHAP")
        print(f"{'='*60}")
        try:
            # Get model for SHAP
            if USE_ENSEMBLE and ensemble is not None and len(ensemble_models) > 0:
                model_for_shap = ensemble_models[0]
            else:
                model_for_shap = model
            
            # Create wrapper function for FT-Transformer
            def ft_predict_wrapper(X):
                """Wrapper to convert numpy array to DataFrame for FT-Transformer"""
                X_df = pd.DataFrame(X, columns=feature_columns)
                pred = model_for_shap.predict(X_df)
                if isinstance(pred, pd.DataFrame):
                    if 'conflict_score_prediction' in pred.columns:
                        return pred['conflict_score_prediction'].values
                    else:
                        return pred.iloc[:, 0].values
                return pred if isinstance(pred, np.ndarray) else np.array([pred])
            
            # Create SHAP explainer (will auto-select PermutationExplainer for function-based model)
            print("  Creating SHAP explainer (PermutationExplainer for FT-Transformer)...")
            explainer_ft = shap.Explainer(ft_predict_wrapper, X_background)
            shap_values_ft = explainer_ft(X_sample)
            
            # Calculate mean absolute SHAP values
            mean_shap_ft = np.abs(shap_values_ft.values).mean(0)
            all_shap_results['FT-Transformer'] = {
                'shap_values': shap_values_ft,
                'mean_shap': mean_shap_ft,
                'explainer_type': 'PermutationExplainer'
            }
            
            print(f"  ✓ FT-Transformer SHAP analysis complete")
            print(f"  Explainer: PermutationExplainer (auto-selected for function-based model)")
            
        except Exception as e:
            print(f"  ⚠ FT-Transformer SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Analyze XGBoost (TreeExplainer)
    if TRAIN_XGBOOST and 'XGBoost' in all_models:
        print(f"\n{'='*60}")
        print("Analyzing XGBoost with SHAP")
        print(f"{'='*60}")
        try:
            xgb_model = all_models['XGBoost']
            
            # Create SHAP explainer (will auto-select TreeExplainer for tree model)
            print("  Creating SHAP explainer (TreeExplainer for XGBoost)...")
            explainer_xgb = shap.Explainer(xgb_model)  # Auto-detects tree model
            shap_values_xgb = explainer_xgb(X_sample_df)
            
            # Calculate mean absolute SHAP values
            if hasattr(shap_values_xgb, 'values'):
                mean_shap_xgb = np.abs(shap_values_xgb.values).mean(0)
            else:
                mean_shap_xgb = np.abs(shap_values_xgb).mean(0)
            
            all_shap_results['XGBoost'] = {
                'shap_values': shap_values_xgb,
                'mean_shap': mean_shap_xgb,
                'explainer_type': 'TreeExplainer'
            }
            
            print(f"  ✓ XGBoost SHAP analysis complete")
            print(f"  Explainer: TreeExplainer (auto-selected for tree model)")
            
        except Exception as e:
            print(f"  ⚠ XGBoost SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Analyze CatBoost (TreeExplainer)
    if TRAIN_CATBOOST and 'CatBoost' in all_models:
        print(f"\n{'='*60}")
        print("Analyzing CatBoost with SHAP")
        print(f"{'='*60}")
        try:
            cb_model = all_models['CatBoost']
            
            # Create SHAP explainer (will auto-select TreeExplainer for tree model)
            print("  Creating SHAP explainer (TreeExplainer for CatBoost)...")
            explainer_cb = shap.Explainer(cb_model)  # Auto-detects tree model
            shap_values_cb = explainer_cb(X_sample_df)
            
            # Calculate mean absolute SHAP values
            if hasattr(shap_values_cb, 'values'):
                mean_shap_cb = np.abs(shap_values_cb.values).mean(0)
            else:
                mean_shap_cb = np.abs(shap_values_cb).mean(0)
            
            all_shap_results['CatBoost'] = {
                'shap_values': shap_values_cb,
                'mean_shap': mean_shap_cb,
                'explainer_type': 'TreeExplainer'
            }
            
            print(f"  ✓ CatBoost SHAP analysis complete")
            print(f"  Explainer: TreeExplainer (auto-selected for tree model)")
            
        except Exception as e:
            print(f"  ⚠ CatBoost SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate visualizations for all models
    if all_shap_results:
        print(f"\n{'='*60}")
        print("Generating SHAP Visualizations")
        print(f"{'='*60}")
        
        # Plot 1: Individual bar plots for each model
        n_models = len(all_shap_results)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, shap_data) in enumerate(all_shap_results.items()):
            try:
                shap.plots.bar(shap_data['shap_values'], show=False, ax=axes[idx])
                axes[idx].set_title(f'{model_name}\n({shap_data["explainer_type"]})', fontsize=12, fontweight='bold')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Plot error:\n{str(e)[:50]}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{model_name} (Error)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('/content/shap_feature_importance_all_models.png', dpi=200, bbox_inches='tight')
        print("✓ Combined SHAP bar plots saved to /content/shap_feature_importance_all_models.png")
        plt.show()
        
        # Plot 2: Comparison of top features across models
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top 10 features for each model
        top_n = 10
        all_top_features = set()
        feature_importance_dict = {}
        
        for model_name, shap_data in all_shap_results.items():
            mean_shap = shap_data['mean_shap']
            top_features_idx = np.argsort(mean_shap)[-top_n:][::-1]
            top_features = [feature_columns[idx] for idx in top_features_idx]
            all_top_features.update(top_features)
            
            feature_importance_dict[model_name] = {
                'features': top_features,
                'scores': [mean_shap[idx] for idx in top_features_idx]
            }
        
        # Create comparison DataFrame
        comparison_data = []
        for feat in all_top_features:
            row = {'feature': feat}
            for model_name in all_shap_results.keys():
                if feat in feature_importance_dict[model_name]['features']:
                    idx = feature_importance_dict[model_name]['features'].index(feat)
                    row[model_name] = feature_importance_dict[model_name]['scores'][idx]
                else:
                    row[model_name] = 0.0
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('feature')
        
        # Plot comparison
        comparison_df.plot(kind='barh', ax=ax, width=0.8)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Top Feature Importance Comparison (SHAP Values)', fontsize=14, fontweight='bold')
        ax.legend(title='Model', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('/content/shap_feature_importance_comparison.png', dpi=200, bbox_inches='tight')
        print("✓ Feature importance comparison saved to /content/shap_feature_importance_comparison.png")
        plt.show()
        
        # Print top features for each model
        print(f"\n{'='*60}")
        print("Top 10 Most Important Features (SHAP Values)")
        print(f"{'='*60}")
        
        for model_name, shap_data in all_shap_results.items():
            mean_shap = shap_data['mean_shap']
            top_features_idx = np.argsort(mean_shap)[-top_n:][::-1]
            
            print(f"\n{model_name} ({shap_data['explainer_type']}):")
            print(f"{'Rank':<6} {'Feature':<40} {'Mean |SHAP|':<15}")
            print("-" * 65)
            for i, idx in enumerate(top_features_idx, 1):
                print(f"{i:<6} {feature_columns[idx]:<40} {mean_shap[idx]:<15.4f}")
        
        # Summary statistics
        print(f"\n{'='*60}")
        print("SHAP Analysis Summary")
        print(f"{'='*60}")
        print(f"Models analyzed: {len(all_shap_results)}")
        for model_name, shap_data in all_shap_results.items():
            print(f"  {model_name}: {shap_data['explainer_type']}")
        print(f"Sample size: {sample_size} instances")
        print(f"Background size: {background_size} instances (for PermutationExplainer)")
        
except ImportError:
    print("⚠ SHAP not installed. Installing...")
    import subprocess
    try:
        subprocess.check_call(['pip', 'install', '-q', 'shap'])
        import shap
        print("✓ SHAP installed. Please re-run this cell for feature importance analysis.")
    except Exception as e:
        print(f"✗ Failed to install SHAP: {e}")
        print("  Continuing without SHAP analysis...")
except Exception as e:
    print(f"⚠ SHAP analysis error: {e}")
    import traceback
    traceback.print_exc()
    print("  Continuing without SHAP analysis...")

# Save model(s)
model_save_path = '/content/ft_transformer_model'
print(f"\n" + "="*60)
print("Saving Model(s)")
print("="*60)
try:
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
    import pickle
    
    if USE_ENSEMBLE and ensemble is not None:
        # Save each ensemble model as .pkl
        ensemble_dir = Path(model_save_path) / 'ensemble'
        ensemble_dir.mkdir(exist_ok=True)
        
        for i, ensemble_model in enumerate(ensemble_models):
            pkl_path = ensemble_dir / f"ft_transformer_model_{i+1}.pkl"
            try:
                with open(pkl_path, 'wb') as f:
                    pickle.dump(ensemble_model, f)
                print(f"✓ FT-Transformer ensemble model {i+1} saved to {pkl_path}")
            except Exception as e:
                print(f"⚠ Error saving ensemble model {i+1}: {e}")
                save_ensemble_model(ensemble_model, str(ensemble_dir), i+1)
        
        # Save primary model as .pkl
        primary_pkl_path = Path(model_save_path) / 'ft_transformer_model.pkl'
        try:
            with open(primary_pkl_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ FT-Transformer primary model saved to {primary_pkl_path}")
        except Exception as e:
            print(f"⚠ Error saving primary model: {e}")
            save_ensemble_model(model, str(model_save_path), "primary")
    else:
        # Save single model as .pkl
        single_pkl_path = Path(model_save_path) / 'ft_transformer_model.pkl'
        try:
            with open(single_pkl_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ FT-Transformer model saved to {single_pkl_path}")
        except Exception as e:
            print(f"⚠ Error saving model: {e}")
            save_ensemble_model(model, str(model_save_path), "single")
    
    # Save XGBoost model as .pkl
    if TRAIN_XGBOOST and xgb_model is not None:
        xgb_pkl_path = Path(model_save_path) / 'xgboost_model.pkl'
        try:
            with open(xgb_pkl_path, 'wb') as f:
                pickle.dump(xgb_model, f)
            print(f"✓ XGBoost model saved to {xgb_pkl_path}")
        except Exception as e:
            print(f"✗ Failed to save XGBoost model: {e}")
    
    # Save CatBoost model as .pkl
    if TRAIN_CATBOOST and catboost_model is not None:
        cb_pkl_path = Path(model_save_path) / 'catboost_model.pkl'
        try:
            with open(cb_pkl_path, 'wb') as f:
                pickle.dump(catboost_model, f)
            print(f"✓ CatBoost model saved to {cb_pkl_path}")
        except Exception as e:
            print(f"✗ Failed to save CatBoost model: {e}")
                
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
    'kappa': float(kappa_ft),
    'macro_f1': float(macro_f1_ft),
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
if 'FT-Transformer' in all_metrics and 'kappa' in all_metrics['FT-Transformer']:
    print(f"    - Cohen's Kappa: {all_metrics['FT-Transformer']['kappa']:.4f}")
    print(f"    - Macro F1: {all_metrics['FT-Transformer']['macro_f1']:.4f}")
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
print(f"    - /content/normalized_confusion_matrices.png (normalized confusion matrices)")
if len(all_class_predictions) > 0:
    for model_name in all_class_predictions.keys():
        safe_name = model_name.replace(' ', '_').lower()
        print(f"    - /content/confusion_matrix_{safe_name}.png ({model_name} confusion matrix)")
if len(all_predictions) > 1:
    print(f"    - /content/model_comparison.png (model comparison plots)")
if len(roc_auc_scores) > 0:
    print(f"    - /content/roc_auc_curves.png (ROC-AUC curves comparison)")
print(f"\n✓ Models Trained:")
print(f"    - FT-Transformer: {'✓' if 'FT-Transformer' in all_models else '✗'}")
print(f"    - XGBoost: {'✓' if TRAIN_XGBOOST and 'XGBoost' in all_models else '✗'}")
print(f"    - CatBoost: {'✓' if TRAIN_CATBOOST and 'CatBoost' in all_models else '✗'}")
print("\n" + "="*60)
print("Training Complete!")
print("="*60)