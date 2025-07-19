def load_model(model_path):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["features_names"]

def predict(model, features):
    return model.predict(features)
def save_model(model, filepath):
    """Save the model to a file."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Load the model from a file."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model

def predict_churn(model, X):
    """Predict churn using the loaded model."""
    return model.predict(X)
import pickle
import argparse
import pandas as pd
import sys

def load_model(model_path):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["features_names"]

def predict(model, features):
    return model.predict(features)

def save_model(model, filepath):
    """Save the model to a file."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model_simple(filepath):
    """Load the model from a file."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model

def predict_churn(model, X):
    """Predict churn using the loaded model."""
    return model.predict(X)

def main():
    parser = argparse.ArgumentParser(description="Predict customer churn using a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file (pickle)")
    parser.add_argument("--data", type=str, required=True, help="Path to the input CSV file with features")
    parser.add_argument("--output", type=str, default=None, help="Path to save predictions (CSV)")
    args = parser.parse_args()

    try:
        model, feature_names = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.data)
        X = df[feature_names]
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        predictions = predict(model, X)
        df['Churn_Prediction'] = predictions
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")
        else:
            print(df[['Churn_Prediction']].head())
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)