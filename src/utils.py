import pickle


def save_model(model, feature_names, filename):
    model_data = {"model": model, "features_names": feature_names}
    with open(filename, "wb") as f:
        pickle.dump(model_data, f)


def load_model(filename):
    with open(filename, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["features_names"]


def load_and_clean_data(file_path):
    """
    Load and clean the customer data from the specified CSV file.

    Parameters:
    - file_path: str, path to the CSV file containing the customer data

    Returns:
    - df: DataFrame, a pandas DataFrame containing the cleaned customer data
    """
    import pandas as pd

    # Load the data
    df = pd.read_csv(file_path)

    # Data cleaning and preprocessing steps
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )  # Convert TotalCharges to numeric
    df.dropna(how="any", inplace=True)  # Drop rows with null values

    # Bin customers based on tenure
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df["tenure_group"] = pd.cut(df.tenure, range(1, 80, 12), right=False, labels=labels)

    # Remove unnecessary columns
    df.drop(columns=["customerID", "tenure"], axis=1, inplace=True)

    return df


def encode_labels(df):
    """
    Perform label encoding on the categorical features of the DataFrame.

    Parameters:
    - df: DataFrame, a pandas DataFrame containing the customer data with
          categorical features to be encoded

    Returns:
    - df: DataFrame, the input DataFrame with label-encoded categorical
          features
    """
    from sklearn.preprocessing import LabelEncoder

    # Encode the target variable 'Churn'
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

    # Get categorical columns
    object_columns = df.select_dtypes(include="object").columns

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Apply label encoding to each categorical column
    for column in object_columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Special handling for 'tenure_group' column
    df["tenure_group"] = df["tenure_group"].astype(str)
    df["tenure_group"] = label_encoder.fit_transform(df["tenure_group"])

    return df


def split_data(df):
    """
    Split the data into training and testing sets.

    Parameters:
    - df: DataFrame, a pandas DataFrame containing the customer data to be split

    Returns:
    - X_train, X_test, y_train, y_test: split data arrays for training and testing
    """
    from sklearn.model_selection import train_test_split

    # Separate features and target variable from the DataFrame
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def handle_imbalance_smoteenn(X_train, y_train):
    """
    Handle class imbalance using SMOTEENN.

    Parameters:
    - X_train: array-like, feature data for training
    - y_train: array-like, target data for training

    Returns:
    - X_resampled, y_resampled: resampled feature and target data with balanced classes
    """
    from imblearn.combine import SMOTEENN

    # Initialize SMOTEENN
    sm = SMOTEENN()

    # Fit and resample the data
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def handle_imbalance_adasyn(X_train, y_train):
    """
    Handle class imbalance using ADASYN.

    Parameters:
    - X_train: array-like, feature data for training
    - y_train: array-like, target data for training

    Returns:
    - X_resampled2, y_resampled2: resampled feature and target data with balanced classes
    """
    from imblearn.over_sampling import ADASYN

    # Initialize ADASYN
    ada = ADASYN()

    # Fit and resample the data
    X_resampled2, y_resampled2 = ada.fit_resample(X_train, y_train)

    return X_resampled2, y_resampled2


def train_models(X_train, y_train):
    """
    Train machine learning models (Random Forest, Decision Tree, XGBoost) on the training data.

    Parameters:
    - X_train: array-like, feature data for training
    - y_train: array-like, target data for training

    Returns:
    - models: dictionary, trained models
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    # Initialize models
    dtc = DecisionTreeClassifier(random_state=42)
    rfc = RandomForestClassifier(random_state=42)
    xg = XGBClassifier(random_state=42)

    # Train models
    dtc.fit(X_train, y_train)
    rfc.fit(X_train, y_train)
    xg.fit(X_train, y_train)

    # Store models in a dictionary
    models = {"Decision Tree": dtc, "Random Forest": rfc, "XGBoost": xg}

    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate the trained models on the test data and print the performance metrics.

    Parameters:
    - models: dictionary, trained models
    - X_test: array-like, feature data for testing
    - y_test: array-like, target data for testing
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Iterate over the models and evaluate each one
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"===== {name} Performance =====")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("\n")


def tune_hyperparameters(models, X_train, y_train):
    """
    Tune hyperparameters of the models using RandomizedSearchCV.

    Parameters:
    - models: dictionary, models to be tuned
    - X_train: array-like, feature data for training
    - y_train: array-like, target data for training

    Returns:
    - best_models: dictionary, models with best hyperparameters
    """
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np

    # Define parameter grids for each model
    xg_params = {
        "n_estimators": np.arange(50, 500, 50),
        "max_depth": np.arange(3, 15, 2),
        "learning_rate": np.linspace(0.01, 0.3, 10),
        "subsample": np.linspace(0.5, 1, 5),
        "colsample_bytree": np.linspace(0.5, 1, 5),
        "gamma": np.linspace(0, 5, 5),
        "reg_lambda": np.logspace(-2, 2, 5),
        "reg_alpha": np.logspace(-2, 2, 5),
        "min_child_weight": np.arange(1, 10, 2),
        "scale_pos_weight": np.linspace(1, 10, 5),
    }

    rfc_params = {
        "n_estimators": np.arange(50, 500, 50),
        "max_depth": np.arange(5, 30, 5),
        "min_samples_split": np.arange(2, 10, 2),
        "min_samples_leaf": np.arange(1, 10, 2),
        "max_features": ["sqrt", "log2"],
    }

    # Initialize RandomizedSearchCV for each model
    xg_random = RandomizedSearchCV(
        models["XGBoost"],
        xg_params,
        n_iter=50,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
    )
    rfc_random = RandomizedSearchCV(
        models["Random Forest"],
        rfc_params,
        n_iter=30,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
    )

    # Fit the models
    xg_random.fit(X_train, y_train)
    rfc_random.fit(X_train, y_train)

    # Get the best models
    best_models = {
        "XGBoost": xg_random.best_estimator_,
        "Random Forest": rfc_random.best_estimator_,
    }

    return best_models


def save_trained_model(model, file_name):
    """
    Save the trained model to a file (simple pickle version).

    Parameters:
    - model: the trained model to be saved
    - file_name: str, the name of the file where the model will be saved
    """
    import pickle

    # Save the model using pickle
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_trained_model(file_name):
    """
    Load a trained model from a file (simple pickle version).

    Parameters:
    - file_name: str, the name of the file from which the model will be loaded

    Returns:
    - model: the loaded model
    """
    import pickle

    # Load the model using pickle
    with open(file_name, "rb") as f:
        model = pickle.load(f)

    return model
