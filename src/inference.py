import pickle

def load_model(model_path):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["features_names"]

def predict(model, features):
    return model.predict(features)
import pickle

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