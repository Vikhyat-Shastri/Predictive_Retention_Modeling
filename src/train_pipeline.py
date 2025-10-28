"""
Retrain the model with improved LightGBM parameters
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from preprocess import load_and_clean
from pipeline import ChurnPredictionPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib

print("="*80)
print("RETRAINING MODEL WITH IMPROVED PARAMETERS")
print("="*80)

# Load and prepare data
print("\n1. Loading data...")
# Use relative path from src directory
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco_customer_churn.csv')
df = load_and_clean(data_path)
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features: {X.shape[1]}")

# Create and train improved pipeline
print("\n2. Training improved LightGBM model...")
pipeline = ChurnPredictionPipeline(
    model_type='lgbm',
    resampling_method='smoteenn'
)

# Train with cross-validation
cv_results = pipeline.train(X_train, y_train, cv_folds=5)

print(f"   Cross-validation complete")
if 'cv_scores' in cv_results:
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        mean_score = cv_results['cv_scores'][metric]['mean']
        std_score = cv_results['cv_scores'][metric]['std']
        print(f"   CV {metric}: {mean_score:.3f} (+/- {std_score:.3f})")
else:
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        mean_score = cv_results[metric]['mean']
        std_score = cv_results[metric]['std']
        print(f"   CV {metric}: {mean_score:.3f} (+/- {std_score:.3f})")

# Evaluate on test set
print("\n3. Evaluating on test set...")
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("IMPROVED MODEL PERFORMANCE")
print("="*80)
print(f"Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
print(f"Recall:    {recall:.3f} ({recall*100:.1f}%)")
print(f"F1-Score:  {f1:.3f} ({f1*100:.1f}%)")
print(f"ROC-AUC:   {roc_auc:.3f} ({roc_auc*100:.1f}%)")
print("="*80)

# Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"                Predicted")
print(f"                No Churn  Churn")
print(f"Actual No Churn    {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"       Churn       {cm[1,0]:4d}    {cm[1,1]:4d}")

# Save the improved model
print("\n4. Saving improved model...")
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_model_pipeline.pkl')
pipeline.save_model(model_path)
print(f"   ✓ Model saved to {model_path}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
