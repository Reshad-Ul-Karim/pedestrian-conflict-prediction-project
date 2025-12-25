# Explainable AI (XAI) Implementation Guide

## Overview

This guide provides a detailed implementation plan for adding Explainable AI (XAI) capabilities to the pedestrian conflict prediction system. The XAI framework will make the model interpretable and rigorous for research publication.

## Why XAI is Critical for This Research

1. **Model Trust**: Understand why the model makes specific predictions
2. **Feature Validation**: Verify that important features align with domain knowledge
3. **Bias Detection**: Identify potential biases in the model
4. **Research Rigor**: Provide evidence-based explanations for predictions
5. **Regulatory Compliance**: Explainable models are required for safety-critical applications

## XAI Architecture

```
src/xai/
├── __init__.py
├── feature_importance.py      # Permutation importance, correlation analysis
├── shap_analysis.py           # SHAP values computation and visualization
├── attention_visualization.py # FT-Transformer attention maps
├── conflict_explanation.py     # Per-prediction explanations
├── uncertainty_analysis.py    # Uncertainty quantification and visualization
├── counterfactual_analysis.py # "What-if" scenario analysis
├── decision_boundary.py        # Decision boundary visualization
└── xai_pipeline.py            # Complete XAI workflow
```

## Implementation Details

### 1. Feature Importance Analysis

**File: `src/xai/feature_importance.py`**

**Purpose**: Identify which features are most important for conflict prediction.

**Methods**:

1. **Permutation Importance**:
   ```python
   from sklearn.inspection import permutation_importance
   
   def compute_permutation_importance(model, X_test, y_test, n_repeats=10):
       """Compute permutation importance for all features"""
       perm_importance = permutation_importance(
           model, X_test, y_test,
           n_repeats=n_repeats,
           random_state=42,
           scoring='neg_mean_squared_error'
       )
       
       # Create DataFrame
       importance_df = pd.DataFrame({
           'feature': X_test.columns,
           'importance_mean': perm_importance.importances_mean,
           'importance_std': perm_importance.importances_std
       }).sort_values('importance_mean', ascending=False)
       
       return importance_df
   ```

2. **Feature Correlation Analysis**:
   ```python
   def analyze_feature_correlations(df, target_col='conflict_score'):
       """Analyze correlations between features and target"""
       correlations = df.corr()[target_col].sort_values(ascending=False)
       
       # Visualize
       plt.figure(figsize=(12, 8))
       sns.heatmap(df.corr(), annot=False, cmap='coolwarm', center=0)
       plt.title('Feature Correlation Matrix')
       plt.tight_layout()
       plt.savefig('outputs/xai_report/feature_correlations.png')
       
       return correlations
   ```

3. **Mutual Information**:
   ```python
   from sklearn.feature_selection import mutual_info_regression
   
   def compute_mutual_information(X, y):
       """Compute mutual information between features and target"""
       mi_scores = mutual_info_regression(X, y, random_state=42)
       
       mi_df = pd.DataFrame({
           'feature': X.columns,
           'mutual_information': mi_scores
       }).sort_values('mutual_information', ascending=False)
       
       return mi_df
   ```

**Integration with Current Codebase**:
- Uses `X_test` from `train_ft_transformer_conflict.py`
- Works with preprocessed features from `ConflictDatasetPreprocessor`
- Outputs rankings compatible with feature names

### 2. SHAP Value Computation

**File: `src/xai/shap_analysis.py`**

**Purpose**: Provide local and global explanations using SHAP values.

**Challenge**: FT-Transformer is not tree-based, so we need Kernel SHAP or Deep SHAP.

**Implementation**:

```python
import shap
import torch

class SHAPAnalyzer:
    def __init__(self, model, preprocessor, X_train_sample, device='cpu'):
        self.model = model
        self.preprocessor = preprocessor
        self.X_train_sample = X_train_sample  # Sample for background
        self.device = device
        
        # Create explainer
        # For tabular models, use Kernel SHAP
        self.explainer = shap.KernelExplainer(
            self._model_predict_wrapper,
            self.X_train_sample
        )
    
    def _model_predict_wrapper(self, X):
        """Wrapper to convert numpy to torch and back"""
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=self.preprocessor.feature_columns)
        
        # Preprocess
        X_processed = self.preprocessor.transform(X_df)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_processed.values).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.predict(X_tensor)
        
        return predictions.cpu().numpy()
    
    def compute_shap_values(self, X_test, n_samples=100):
        """Compute SHAP values for test set"""
        shap_values = self.explainer.shap_values(
            X_test.iloc[:n_samples],
            nsamples=100  # Number of samples for Kernel SHAP
        )
        return shap_values
    
    def visualize_summary(self, shap_values, X_test, output_path):
        """Create SHAP summary plot"""
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=X_test.columns,
            show=False
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_waterfall(self, shap_values, X_test, instance_idx, output_path):
        """Create SHAP waterfall plot for single instance"""
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[instance_idx],
                base_values=self.explainer.expected_value,
                data=X_test.iloc[instance_idx],
                feature_names=X_test.columns
            ),
            show=False
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
```

**Integration Points**:
- Uses trained model from `FTTransformerTrainer`
- Uses preprocessor from `ConflictDatasetPreprocessor`
- Works with test data from train/test split

### 3. Attention Visualization

**File: `src/xai/attention_visualization.py`**

**Purpose**: Visualize attention patterns in FT-Transformer to understand feature interactions.

**Challenge**: PyTorch Tabular may not expose attention weights directly. We need to hook into the model.

**Implementation**:

```python
class AttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        # Access transformer layers
        # Note: This depends on PyTorch Tabular's internal structure
        transformer_layers = self.model.model.transformer.layers
        
        for i, layer in enumerate(transformer_layers):
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output is typically (hidden_states, attention_weights)
                    if isinstance(output, tuple) and len(output) > 1:
                        self.attention_weights.append({
                            'layer': layer_idx,
                            'attention': output[1].cpu().numpy()
                        })
                return hook
            
            layer.self_attn.register_forward_hook(make_hook(i))
    
    def extract_attention(self, X_sample):
        """Extract attention weights for a sample"""
        self.attention_weights = []
        
        # Forward pass
        with torch.no_grad():
            _ = self.model.predict(X_sample)
        
        return self.attention_weights
    
    def visualize_attention_heatmap(self, attention_weights, feature_names, output_path):
        """Create attention heatmap"""
        # Average attention across heads
        avg_attention = np.mean(attention_weights, axis=0)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            avg_attention,
            xticklabels=feature_names,
            yticklabels=feature_names,
            cmap='viridis',
            annot=False
        )
        plt.title('Feature Attention Patterns')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
```

**Alternative Approach**: If attention weights are not accessible, use gradient-based attention:
```python
def compute_gradient_attention(model, X, target):
    """Compute attention using gradients (gradient-based attention)"""
    X.requires_grad = True
    output = model(X)
    output.backward()
    
    # Attention is proportional to gradient magnitude
    attention = torch.abs(X.grad)
    return attention
```

### 4. Conflict Score Explanation

**File: `src/xai/conflict_explanation.py`**

**Purpose**: Generate human-readable explanations for individual predictions.

**Implementation**:

```python
class ConflictExplainer:
    def __init__(self, model, preprocessor, shap_analyzer):
        self.model = model
        self.preprocessor = preprocessor
        self.shap_analyzer = shap_analyzer
    
    def explain_prediction(self, features_dict, image_path=None):
        """
        Generate explanation for a single prediction
        
        Args:
            features_dict: Dictionary of feature values
            image_path: Optional path to image for visualization
        
        Returns:
            Dictionary with explanation
        """
        # Convert to DataFrame
        X = pd.DataFrame([features_dict])
        
        # Preprocess
        X_processed = self.preprocessor.transform(X)
        
        # Predict
        prediction = self.model.predict(X_processed)
        risk_level = self._get_risk_level(prediction[0])
        
        # Get SHAP values
        shap_values = self.shap_analyzer.compute_shap_values(X)
        
        # Get top contributing features
        top_features = self._get_top_contributing_features(
            shap_values[0],
            X.columns,
            n=10
        )
        
        # Generate natural language explanation
        explanation_text = self._generate_explanation(
            prediction[0],
            risk_level,
            top_features,
            features_dict
        )
        
        return {
            'predicted_score': float(prediction[0]),
            'risk_level': risk_level,
            'top_contributing_features': top_features,
            'explanation': explanation_text,
            'shap_values': shap_values[0].tolist()
        }
    
    def _get_risk_level(self, score):
        """Convert score to risk level"""
        if score > 0.65:
            return 'HIGH'
        elif score > 0.35:
            return 'MED'
        else:
            return 'LOW'
    
    def _get_top_contributing_features(self, shap_values, feature_names, n=10):
        """Get top N contributing features"""
        contributions = list(zip(feature_names, shap_values))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions[:n]
    
    def _generate_explanation(self, score, risk_level, top_features, features):
        """Generate natural language explanation"""
        explanation = f"Predicted conflict risk: {risk_level} (score: {score:.3f})\n\n"
        explanation += "Key contributing factors:\n"
        
        for i, (feature, contribution) in enumerate(top_features[:5], 1):
            direction = "increases" if contribution > 0 else "decreases"
            explanation += f"{i}. {feature}: {direction} risk by {abs(contribution):.3f}\n"
            # Add context based on feature value
            if feature in features:
                explanation += f"   (Current value: {features[feature]:.3f})\n"
        
        return explanation
    
    def visualize_explanation(self, explanation, image_path, output_path):
        """Create visualization with explanation overlay"""
        if image_path and Path(image_path).exists():
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img_rgb)
            ax.axis('off')
            
            # Add explanation text
            explanation_text = explanation['explanation']
            ax.text(
                0.02, 0.98,
                explanation_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
```

**Integration with `visualize_conflict_risk.py`**:
- Can be called after `compute_conflict_risk()`
- Uses same feature extraction pipeline
- Can overlay explanations on visualization

### 5. Uncertainty Analysis

**File: `src/xai/uncertainty_analysis.py`**

**Purpose**: Visualize and analyze prediction uncertainty from ensemble.

**Implementation**:

```python
class UncertaintyAnalyzer:
    def __init__(self, ensemble_model):
        self.ensemble = ensemble_model
    
    def analyze_uncertainty(self, dataloader):
        """Analyze uncertainty across test set"""
        predictions = self.ensemble.predict_with_uncertainty(dataloader)
        
        return {
            'mean_predictions': predictions['conflict_score_prediction'],
            'std_predictions': predictions['prediction_std'],
            'lower_bounds': predictions['prediction_lower_bound'],
            'upper_bounds': predictions['prediction_upper_bound']
        }
    
    def visualize_uncertainty(self, y_true, predictions_dict, output_path):
        """Create uncertainty visualization"""
        mean_preds = predictions_dict['mean_predictions']
        std_preds = predictions_dict['std_predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Prediction vs. Actual with uncertainty bands
        axes[0, 0].scatter(y_true, mean_preds, alpha=0.5)
        axes[0, 0].errorbar(
            y_true, mean_preds,
            yerr=std_preds,
            fmt='none',
            alpha=0.3
        )
        axes[0, 0].plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        axes[0, 0].set_xlabel('True Conflict Score')
        axes[0, 0].set_ylabel('Predicted Conflict Score')
        axes[0, 0].set_title('Predictions with Uncertainty Bands')
        axes[0, 0].legend()
        
        # 2. Uncertainty distribution
        axes[0, 1].hist(std_preds, bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Prediction Uncertainty (Std)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Uncertainty Distribution')
        
        # 3. Error vs. Uncertainty
        errors = np.abs(y_true - mean_preds)
        axes[1, 0].scatter(std_preds, errors, alpha=0.5)
        axes[1, 0].set_xlabel('Uncertainty (Std)')
        axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].set_title('Uncertainty vs. Error')
        
        # 4. Calibration plot
        # Bin predictions by uncertainty
        uncertainty_bins = np.quantile(std_preds, [0, 0.25, 0.5, 0.75, 1.0])
        bin_errors = []
        for i in range(len(uncertainty_bins) - 1):
            mask = (std_preds >= uncertainty_bins[i]) & (std_preds < uncertainty_bins[i+1])
            bin_errors.append(errors[mask].mean())
        
        axes[1, 1].bar(range(len(bin_errors)), bin_errors)
        axes[1, 1].set_xlabel('Uncertainty Bin')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Calibration: Uncertainty vs. Error')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
```

**Integration with EnsembleModel**:
- Uses `EnsembleModel` from `train_ft_transformer_conflict.py`
- Works with ensemble predictions
- Validates uncertainty calibration

### 6. Counterfactual Analysis

**File: `src/xai/counterfactual_analysis.py`**

**Purpose**: Generate "what-if" scenarios to understand model behavior.

**Implementation**:

```python
from scipy.optimize import minimize

class CounterfactualAnalyzer:
    def __init__(self, model, preprocessor, feature_bounds):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_bounds = feature_bounds  # Dict of (min, max) for each feature
    
    def generate_counterfactual(self, original_features, target_score, 
                                immutable_features=None):
        """
        Find minimal changes to achieve target_score
        
        Args:
            original_features: Original feature vector
            target_score: Desired conflict score
            immutable_features: List of features that cannot be changed
        
        Returns:
            counterfactual_features, changes
        """
        original_array = np.array([original_features])
        original_df = pd.DataFrame(original_array, columns=self.preprocessor.feature_columns)
        original_processed = self.preprocessor.transform(original_df).values[0]
        
        def objective(x):
            """Minimize distance while achieving target"""
            x_df = pd.DataFrame([x], columns=self.preprocessor.feature_columns)
            x_processed = self.preprocessor.transform(x_df)
            pred = self.model.predict(x_processed)[0]
            
            # Distance penalty
            distance = np.sum((x - original_processed) ** 2)
            
            # Target penalty
            target_penalty = (pred - target_score) ** 2
            
            return distance + 10 * target_penalty
        
        # Bounds
        bounds = []
        for col in self.preprocessor.feature_columns:
            if col in self.feature_bounds:
                bounds.append(self.feature_bounds[col])
            else:
                # Default bounds from data
                bounds.append((original_processed.min(), original_processed.max()))
        
        # Constraints: immutable features
        constraints = []
        if immutable_features:
            for i, col in enumerate(self.preprocessor.feature_columns):
                if col in immutable_features:
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda x, idx=i: x[idx] - original_processed[idx]
                    })
        
        # Optimize
        result = minimize(
            objective,
            original_processed,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints if constraints else None
        )
        
        counterfactual_processed = result.x
        
        # Convert back to original feature space
        counterfactual_df = pd.DataFrame(
            [counterfactual_processed],
            columns=self.preprocessor.feature_columns
        )
        counterfactual_features = self.preprocessor.inverse_transform(counterfactual_df).iloc[0]
        
        changes = counterfactual_features - original_features
        
        return counterfactual_features, changes
    
    def analyze_sensitivity(self, original_features, feature_name, 
                           value_range, n_points=20):
        """Analyze how changing one feature affects prediction"""
        values = np.linspace(value_range[0], value_range[1], n_points)
        predictions = []
        
        for val in values:
            modified_features = original_features.copy()
            modified_features[feature_name] = val
            
            modified_df = pd.DataFrame([modified_features], 
                                      columns=self.preprocessor.feature_columns)
            modified_processed = self.preprocessor.transform(modified_df)
            pred = self.model.predict(modified_processed)[0]
            predictions.append(pred)
        
        return values, predictions
```

### 7. Complete XAI Pipeline

**File: `src/xai/xai_pipeline.py`**

**Purpose**: Orchestrate all XAI analyses.

**Implementation**:

```python
class XAIPipeline:
    def __init__(self, model, preprocessor, X_train, X_test, y_test, 
                 feature_names, output_dir='outputs/xai_report/'):
        self.model = model
        self.preprocessor = preprocessor
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.feature_importance_analyzer = FeatureImportanceAnalyzer()
        self.shap_analyzer = SHAPAnalyzer(model, preprocessor, X_train.sample(100))
        self.attention_visualizer = AttentionVisualizer(model)
        self.conflict_explainer = ConflictExplainer(model, preprocessor, self.shap_analyzer)
        self.uncertainty_analyzer = UncertaintyAnalyzer(model)  # If ensemble
    
    def run_complete_analysis(self):
        """Run all XAI analyses"""
        print("=" * 60)
        print("Running Complete XAI Analysis")
        print("=" * 60)
        
        results = {}
        
        # 1. Feature Importance
        print("\n1. Computing feature importance...")
        results['feature_importance'] = self.feature_importance_analyzer.analyze(
            self.model, self.X_test, self.y_test
        )
        
        # 2. SHAP Analysis
        print("2. Computing SHAP values...")
        shap_values = self.shap_analyzer.compute_shap_values(self.X_test.iloc[:100])
        results['shap_values'] = shap_values
        self.shap_analyzer.visualize_summary(
            shap_values, self.X_test.iloc[:100],
            self.output_dir / 'shap_summary.png'
        )
        
        # 3. Attention Visualization
        print("3. Extracting attention patterns...")
        attention_weights = self.attention_visualizer.extract_attention(
            self.X_test.iloc[:10]
        )
        results['attention'] = attention_weights
        
        # 4. Sample Explanations
        print("4. Generating sample explanations...")
        sample_explanations = []
        for idx in range(min(10, len(self.X_test))):
            features_dict = self.X_test.iloc[idx].to_dict()
            explanation = self.conflict_explainer.explain_prediction(features_dict)
            sample_explanations.append(explanation)
        results['explanations'] = sample_explanations
        
        # 5. Uncertainty Analysis (if ensemble)
        if hasattr(self.model, 'ensemble'):
            print("5. Analyzing uncertainty...")
            uncertainty_results = self.uncertainty_analyzer.analyze_uncertainty(
                self.X_test
            )
            results['uncertainty'] = uncertainty_results
        
        # Save results
        self._save_results(results)
        
        print(f"\n✓ XAI analysis complete. Results saved to {self.output_dir}")
        return results
    
    def generate_report(self, results):
        """Generate comprehensive XAI report"""
        # Create HTML/PDF report with all visualizations and analyses
        # Implementation details...
        pass
```

## Integration with Existing Codebase

### Key Integration Points:

1. **Model Loading**:
   ```python
   # From train_ft_transformer_conflict.py
   model = TabularModel.load_from_checkpoint('outputs/models/ft_transformer/best_model.ckpt')
   ```

2. **Preprocessor Loading**:
   ```python
   # From ConflictDatasetPreprocessor
   with open('outputs/models/ft_transformer/preprocessor.pkl', 'rb') as f:
       preprocessor_data = pickle.load(f)
   preprocessor = ConflictDatasetPreprocessor(...)
   preprocessor.scaler = preprocessor_data['scaler']
   preprocessor.feature_columns = preprocessor_data['feature_columns']
   ```

3. **Data Loading**:
   ```python
   # From train_ft_transformer_conflict.py train/test split
   data = preprocessor.load_and_preprocess()
   X_train = data['X_train']
   X_test = data['X_test']
   y_test = data['y_test']
   ```

## Research Questions XAI Answers

1. **Which features are most important?**
   - Feature importance analysis
   - SHAP global importance

2. **Why did the model predict HIGH risk for this person?**
   - SHAP local explanations
   - Conflict score breakdown

3. **How do features interact?**
   - Attention visualization
   - SHAP interaction values

4. **Is the model uncertain about this prediction?**
   - Uncertainty quantification
   - Confidence intervals

5. **What would change the prediction?**
   - Counterfactual analysis
   - Sensitivity analysis

6. **Are there biases in the model?**
   - Feature importance across subgroups
   - Counterfactual fairness analysis

## Expected Outputs

1. **Feature Importance Rankings**: Top 20 most important features
2. **SHAP Summary Plot**: Global feature importance with interaction effects
3. **SHAP Waterfall Plots**: Local explanations for sample predictions
4. **Attention Heatmaps**: Feature interaction patterns
5. **Uncertainty Visualizations**: Prediction intervals and calibration
6. **Counterfactual Examples**: "What-if" scenarios
7. **Comprehensive Report**: PDF/HTML with all analyses

## Next Steps

1. **Create XAI module structure**:
   ```bash
   mkdir -p src/xai
   touch src/xai/__init__.py
   # Create all module files
   ```

2. **Install XAI dependencies**:
   ```bash
   pip install shap matplotlib seaborn scipy
   ```

3. **Implement modules incrementally**:
   - Start with feature importance (easiest)
   - Then SHAP analysis
   - Then attention visualization
   - Finally counterfactual analysis

4. **Test on trained model**:
   ```bash
   python src/xai/xai_pipeline.py \
       --model outputs/models/ft_transformer/best_model.ckpt \
       --data outputs/conflict_dataset.csv \
       --output outputs/xai_report/
   ```

5. **Generate research figures**:
   - Use outputs for paper figures
   - Create publication-quality visualizations

## References

1. **SHAP**: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
2. **Attention**: Vaswani et al. (2017), "Attention Is All You Need"
3. **Counterfactuals**: Wachter et al. (2017), "Counterfactual Explanations"
4. **Uncertainty**: Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"

---

**Status**: Implementation Guide - Ready for Development

