"""
Model Explainability using SHAP (SHapley Additive exPlanations)
Provides interpretable explanations for individual predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import logging
from typing import Any, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    SHAP-based model explainability for churn prediction
    """
    
    def __init__(self, model, X_train: pd.DataFrame = None):
        """
        Initialize explainer
        
        Args:
            model: Trained model or pipeline
            X_train: Training data for background samples (optional)
        """
        self.model = model
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.feature_names = None
        self.X_background = X_train
        
        # Initialize matplotlib for SHAP plots
        shap.initjs()
    
    def _get_model_from_pipeline(self, pipeline):
        """Extract the actual model from a sklearn pipeline"""
        if hasattr(pipeline, 'named_steps'):
            # It's a pipeline
            if 'classifier' in pipeline.named_steps:
                return pipeline.named_steps['classifier']
            else:
                # Get the last step
                return pipeline.steps[-1][1]
        return pipeline
    
    def _preprocess_data(self, X: pd.DataFrame):
        """Preprocess data through pipeline if needed"""
        if hasattr(self.model, 'named_steps'):
            # It's a pipeline - transform through all steps except final classifier
            X_transformed = X.copy()
            for name, transformer in self.model.steps[:-1]:
                X_transformed = transformer.transform(X_transformed)
            return X_transformed
        return X
    
    def create_explainer(self, method: str = 'tree', X_background: pd.DataFrame = None):
        """
        Create SHAP explainer
        
        Args:
            method: 'tree', 'kernel', or 'linear'
            X_background: Background data for KernelExplainer (optional)
        """
        logger.info(f"Creating SHAP {method} explainer...")
        
        actual_model = self._get_model_from_pipeline(self.model)
        
        if method == 'tree':
            # For tree-based models (RF, XGBoost, LightGBM)
            self.explainer = shap.TreeExplainer(actual_model)
            logger.info("TreeExplainer created successfully")
            
        elif method == 'kernel':
            # For any model - slower but model-agnostic
            if X_background is None and self.X_background is not None:
                X_background = self.X_background
            
            if X_background is None:
                raise ValueError("Background data required for KernelExplainer")
            
            # Use a sample for faster computation
            background_sample = shap.sample(X_background, min(100, len(X_background)))
            
            def model_predict(X):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)[:, 1]
                return self.model.predict(X)
            
            self.explainer = shap.KernelExplainer(model_predict, background_sample)
            logger.info("KernelExplainer created successfully")
            
        elif method == 'linear':
            # For linear models
            self.explainer = shap.LinearExplainer(actual_model, X_background)
            logger.info("LinearExplainer created successfully")
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def explain_instance(self, X: pd.DataFrame, index: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction
        
        Args:
            X: Input data
            index: Index of instance to explain
            
        Returns:
            Dictionary with explanation data
        """
        if self.explainer is None:
            self.create_explainer()
        
        # Get single instance
        instance = X.iloc[index:index+1]
        
        # Calculate SHAP values
        if isinstance(self.explainer, shap.TreeExplainer):
            # For tree explainer, we need transformed data
            actual_model = self._get_model_from_pipeline(self.model)
            X_transformed = self._preprocess_data(instance)
            shap_values = self.explainer.shap_values(X_transformed)
            
            # Handle multi-output (classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get positive class
        else:
            shap_values = self.explainer.shap_values(instance)
        
        # Get prediction
        prediction = self.model.predict(instance)[0]
        prediction_proba = self.model.predict_proba(instance)[0] if hasattr(self.model, 'predict_proba') else None
        
        # Get feature names
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Create feature importance dict
        if len(shap_values.shape) == 1:
            feature_contributions = dict(zip(feature_names, shap_values))
        else:
            feature_contributions = dict(zip(feature_names, shap_values[0]))
        
        # Sort by absolute value
        feature_contributions = dict(sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return {
            'prediction': prediction,
            'prediction_proba': prediction_proba,
            'shap_values': shap_values,
            'feature_contributions': feature_contributions,
            'expected_value': self.explainer.expected_value,
            'feature_names': feature_names,
            'instance_values': instance.iloc[0].to_dict()
        }
    
    def plot_force_plot(self, explanation: Dict[str, Any], save_path: str = None):
        """
        Create SHAP force plot for a single prediction
        
        Args:
            explanation: Output from explain_instance()
            save_path: Path to save plot (optional)
        """
        shap_values = explanation['shap_values']
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        expected_value = explanation['expected_value']
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        feature_names = explanation['feature_names']
        instance_values = list(explanation['instance_values'].values())
        
        # Create force plot
        shap.force_plot(
            expected_value,
            shap_values,
            instance_values,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_waterfall(self, explanation: Dict[str, Any], save_path: str = None):
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            explanation: Output from explain_instance()
            save_path: Path to save plot (optional)
        """
        shap_values = explanation['shap_values']
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        expected_value = explanation['expected_value']
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        # Create Explanation object for waterfall plot
        shap_explanation = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=list(explanation['instance_values'].values()),
            feature_names=explanation['feature_names']
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_explanation, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for entire dataset
        
        Args:
            X: Input data
            
        Returns:
            Array of SHAP values
        """
        if self.explainer is None:
            self.create_explainer()
        
        logger.info("Calculating SHAP values for dataset...")
        
        if isinstance(self.explainer, shap.TreeExplainer):
            actual_model = self._get_model_from_pipeline(self.model)
            X_transformed = self._preprocess_data(X)
            shap_values = self.explainer.shap_values(X_transformed)
        else:
            shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class
        
        self.shap_values = shap_values
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        logger.info("SHAP values calculated successfully")
        return shap_values
    
    def plot_summary(self, X: pd.DataFrame = None, shap_values: np.ndarray = None, 
                    save_path: str = None, plot_type: str = 'dot'):
        """
        Create SHAP summary plot
        
        Args:
            X: Input data
            shap_values: Precomputed SHAP values (optional)
            save_path: Path to save plot (optional)
            plot_type: 'dot', 'bar', or 'violin'
        """
        if shap_values is None:
            if self.shap_values is None:
                if X is None:
                    raise ValueError("Either X or shap_values must be provided")
                shap_values = self.calculate_shap_values(X)
            else:
                shap_values = self.shap_values
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X, plot_type='bar', show=False)
        else:
            shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_dependence(self, feature: str, X: pd.DataFrame, interaction_feature: str = None,
                       shap_values: np.ndarray = None, save_path: str = None):
        """
        Create SHAP dependence plot
        
        Args:
            feature: Feature to plot
            X: Input data
            interaction_feature: Feature to use for coloring (optional)
            shap_values: Precomputed SHAP values (optional)
            save_path: Path to save plot (optional)
        """
        if shap_values is None:
            if self.shap_values is None:
                shap_values = self.calculate_shap_values(X)
            else:
                shap_values = self.shap_values
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            shap_values,
            X,
            interaction_index=interaction_feature,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dependence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(self, X: pd.DataFrame = None, 
                              shap_values: np.ndarray = None) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values
        
        Args:
            X: Input data
            shap_values: Precomputed SHAP values (optional)
            
        Returns:
            DataFrame with feature importance
        """
        if shap_values is None:
            if self.shap_values is None:
                if X is None:
                    raise ValueError("Either X or shap_values must be provided")
                shap_values = self.calculate_shap_values(X)
            else:
                shap_values = self.shap_values
        
        feature_names = X.columns.tolist() if X is not None and hasattr(X, 'columns') else self.feature_names
        
        # Calculate mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def create_explanation_report(self, X: pd.DataFrame, index: int = 0, 
                                 save_dir: str = '../models/explanations/') -> Dict[str, Any]:
        """
        Create comprehensive explanation report for a single instance
        
        Args:
            X: Input data
            index: Index of instance to explain
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with all explanation data
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Get explanation
        explanation = self.explain_instance(X, index)
        
        # Create plots
        self.plot_force_plot(explanation, f"{save_dir}force_plot_{index}.png")
        self.plot_waterfall(explanation, f"{save_dir}waterfall_plot_{index}.png")
        
        # Create text report
        report = f"""
CHURN PREDICTION EXPLANATION REPORT
{'='*60}

Instance Index: {index}

PREDICTION:
-----------
Predicted Class: {'CHURN' if explanation['prediction'] == 1 else 'NO CHURN'}
Churn Probability: {explanation['prediction_proba'][1]*100:.2f}%
No-Churn Probability: {explanation['prediction_proba'][0]*100:.2f}%

TOP 10 MOST INFLUENTIAL FEATURES:
----------------------------------
"""
        
        for i, (feature, contribution) in enumerate(list(explanation['feature_contributions'].items())[:10], 1):
            direction = "increases" if contribution > 0 else "decreases"
            impact = "CHURN risk" if contribution > 0 else "RETENTION likelihood"
            value = explanation['instance_values'][feature]
            report += f"{i}. {feature} = {value}\n"
            report += f"   SHAP value: {contribution:.4f} ({direction} {impact})\n\n"
        
        # Save report
        with open(f"{save_dir}explanation_report_{index}.txt", 'w') as f:
            f.write(report)
        
        logger.info(f"Explanation report created for instance {index}")
        
        return {
            'explanation': explanation,
            'report_text': report,
            'plots': {
                'force_plot': f"{save_dir}force_plot_{index}.png",
                'waterfall_plot': f"{save_dir}waterfall_plot_{index}.png"
            }
        }
    
    def save_explainer(self, filepath: str):
        """Save explainer object"""
        explainer_data = {
            'explainer': self.explainer,
            'shap_values': self.shap_values,
            'feature_names': self.feature_names,
            'expected_value': self.expected_value
        }
        joblib.dump(explainer_data, filepath)
        logger.info(f"Explainer saved to {filepath}")
    
    @classmethod
    def load_explainer(cls, filepath: str, model):
        """Load saved explainer"""
        explainer_data = joblib.load(filepath)
        
        instance = cls(model)
        instance.explainer = explainer_data['explainer']
        instance.shap_values = explainer_data['shap_values']
        instance.feature_names = explainer_data['feature_names']
        instance.expected_value = explainer_data['expected_value']
        
        logger.info(f"Explainer loaded from {filepath}")
        return instance


if __name__ == "__main__":
    # Example usage
    from preprocess import load_and_clean
    from pipeline import ChurnPredictionPipeline
    from sklearn.model_selection import train_test_split
    
    # Load data
    df = load_and_clean('../data/Telco_customer_churn.csv')
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load or train model
    try:
        pipeline = ChurnPredictionPipeline.load_model('../models/churn_model_pipeline.pkl')
    except:
        pipeline = ChurnPredictionPipeline(model_type='lgbm', resampling_method='smoteenn')
        pipeline.train(X_train, y_train)
        pipeline.save_model('../models/churn_model_pipeline.pkl')
    
    # Create explainer
    explainer = ModelExplainer(pipeline.pipeline, X_train)
    explainer.create_explainer(method='tree')
    
    # Calculate SHAP values
    shap_values = explainer.calculate_shap_values(X_test)
    
    # Create summary plots
    explainer.plot_summary(X_test, save_path='../models/shap_summary.png')
    explainer.plot_summary(X_test, save_path='../models/shap_importance.png', plot_type='bar')
    
    # Get feature importance
    importance = explainer.get_feature_importance(X_test)
    print("\nFeature Importance (Mean |SHAP|):")
    print(importance.head(10))
    
    # Explain specific instances
    for i in [0, 1, 2]:
        report = explainer.create_explanation_report(X_test, index=i)
        print(report['report_text'])
    
    # Save explainer
    explainer.save_explainer('../models/shap_explainer.pkl')
