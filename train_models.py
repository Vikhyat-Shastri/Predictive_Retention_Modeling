"""
Complete Model Training Script
Trains all models: Pipeline, Segmentation, and Explainability
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_and_clean
from pipeline import ChurnPredictionPipeline, train_multiple_models
from segmentation import CustomerSegmentation
from explainability import ModelExplainer
from sklearn.model_selection import train_test_split
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    
    logger.info("="*80)
    logger.info("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    logger.info("="*80)
    
    # Step 1: Load and preprocess data
    logger.info("\n Step 1: Loading and preprocessing data...")
    try:
        df = load_and_clean('data/Telco_customer_churn.csv')
        logger.info(f"✓ Data loaded successfully: {df.shape}")
        logger.info(f"  - Features: {df.shape[1]}")
        logger.info(f"  - Samples: {df.shape[0]}")
        logger.info(f"  - Churn rate: {df['Churn'].mean()*100:.2f}%")
    except Exception as e:
        logger.error(f"✗ Failed to load data: {e}")
        return
    
    # Step 2: Train prediction models
    logger.info("\n"+"="*80)
    logger.info("Step 2: Training prediction models...")
    logger.info("="*80)
    
    try:
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple models for comparison
        results = train_multiple_models(df, test_size=0.2)
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_scores']['f1'])
        best_pipeline = results[best_model_name]['pipeline']
        
        logger.info(f"\n✓ Best Model: {best_model_name}")
        logger.info(f"  - Accuracy: {results[best_model_name]['test_scores']['accuracy']:.4f}")
        logger.info(f"  - Precision: {results[best_model_name]['test_scores']['precision']:.4f}")
        logger.info(f"  - Recall: {results[best_model_name]['test_scores']['recall']:.4f}")
        logger.info(f"  - F1-Score: {results[best_model_name]['test_scores']['f1']:.4f}")
        logger.info(f"  - ROC-AUC: {results[best_model_name]['test_scores']['roc_auc']:.4f}")
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        best_pipeline.save_model('models/churn_model_pipeline.pkl')
        logger.info("✓ Model saved: models/churn_model_pipeline.pkl")
        
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Step 3: Customer Segmentation
    logger.info("\n"+"="*80)
    logger.info("Step 3: Customer Segmentation (K-Means)...")
    logger.info("="*80)
    
    try:
        segmenter = CustomerSegmentation(n_clusters=4, random_state=42)
        
        # Find optimal clusters
        X_seg = segmenter.prepare_features(df)
        metrics = segmenter.find_optimal_clusters(X_seg, max_clusters=8)
        logger.info("✓ Optimal cluster analysis completed")
        
        # Fit segmentation model
        segmenter.fit(df)
        logger.info("✓ Segmentation model trained")
        
        # Create profiles
        profiles = segmenter.create_segment_profiles(df)
        logger.info(f"✓ Created {len(profiles)} customer segments")
        
        # Visualize
        segmenter.visualize_segments(df, save_path='models/segments_visualization.png')
        logger.info("✓ Segmentation visualizations created")
        
        # Generate insights
        insights = segmenter.generate_segment_insights(df)
        
        # Save insights
        with open('models/segment_insights.txt', 'w') as f:
            for segment_id, insight in insights.items():
                f.write(insight + "\n\n")
        logger.info("✓ Segment insights saved: models/segment_insights.txt")
        
        # Save model
        segmenter.save_model('models/segmentation_model.pkl')
        logger.info("✓ Segmentation model saved: models/segmentation_model.pkl")
        
    except Exception as e:
        logger.error(f"✗ Segmentation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Step 4: Model Explainability (SHAP)
    logger.info("\n"+"="*80)
    logger.info("Step 4: Model Explainability (SHAP)...")
    logger.info("="*80)
    
    try:
        explainer = ModelExplainer(best_pipeline.pipeline, X_train)
        explainer.create_explainer(method='tree')
        logger.info("✓ SHAP explainer created")
        
        # Calculate SHAP values
        shap_values = explainer.calculate_shap_values(X_test)
        logger.info("✓ SHAP values calculated")
        
        # Create summary plots
        explainer.plot_summary(X_test, save_path='models/shap_summary.png')
        explainer.plot_summary(X_test, save_path='models/shap_importance.png', plot_type='bar')
        logger.info("✓ SHAP visualizations created")
        
        # Get feature importance
        importance = explainer.get_feature_importance(X_test)
        importance.to_csv('models/feature_importance.csv', index=False)
        logger.info("✓ Feature importance saved: models/feature_importance.csv")
        
        # Create example explanations
        os.makedirs('models/explanations', exist_ok=True)
        for i in range(min(3, len(X_test))):
            report = explainer.create_explanation_report(
                X_test, 
                index=i,
                save_dir='models/explanations/'
            )
        logger.info("✓ Example explanations created: models/explanations/")
        
        # Save explainer
        explainer.save_explainer('models/shap_explainer.pkl')
        logger.info("✓ SHAP explainer saved: models/shap_explainer.pkl")
        
    except Exception as e:
        logger.error(f"✗ Explainability setup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Final Summary
    logger.info("\n"+"="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info("\nGenerated Files:")
    logger.info("  - models/churn_model_pipeline.pkl")
    logger.info("  - models/segmentation_model.pkl")
    logger.info("  - models/shap_explainer.pkl")
    logger.info("  - models/segments_visualization.png")
    logger.info("  - models/shap_summary.png")
    logger.info("  - models/shap_importance.png")
    logger.info("  - models/feature_importance.csv")
    logger.info("  - models/segment_insights.txt")
    logger.info("  - models/explanations/")
    logger.info("\nNext Steps:")
    logger.info("  1. Review training logs and metrics")
    logger.info("  2. Run the Streamlit app: streamlit run app/streamlit_app.py")
    logger.info("  3. Start the API: python app/api.py")
    logger.info("  4. Or use Docker: docker-compose up")
    logger.info("\n"+"="*80)


if __name__ == "__main__":
    main()
