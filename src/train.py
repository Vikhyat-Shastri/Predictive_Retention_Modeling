import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN

def split_and_resample(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    ada = ADASYN()
    X_resampled2, y_resampled2 = ada.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, X_resampled, y_resampled, X_resampled2, y_resampled2

def train_models(X_train, y_train):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = {"scores": scores, "mean": scores.mean(), "std": scores.std()}
    return results

def tune_xgboost(X, y, base_model):
    xg_params = {
        'n_estimators': np.arange(50, 500, 50),
        'max_depth': np.arange(3, 15, 2),
        'learning_rate': np.linspace(0.01, 0.3, 10),
        'subsample': np.linspace(0.5, 1, 5),
        'colsample_bytree': np.linspace(0.5, 1, 5),
        'gamma': np.linspace(0, 5, 5),
        'reg_lambda': np.logspace(-2, 2, 5),
        'reg_alpha': np.logspace(-2, 2, 5),
        'min_child_weight': np.arange(1, 10, 2),
        'scale_pos_weight': np.linspace(1, 10, 5)
    }
    xg_random = RandomizedSearchCV(base_model, xg_params, n_iter=50, cv=5, scoring='f1', n_jobs=-1, random_state=42)
    xg_random.fit(X, y)
    return xg_random.best_estimator_, xg_random.best_params_

def tune_rf(X, y, base_model):
    rfc_params = {
        'n_estimators': np.arange(50, 500, 50),
        'max_depth': np.arange(5, 30, 5),
        'min_samples_split': np.arange(2, 10, 2),
        'min_samples_leaf': np.arange(1, 10, 2),
        'max_features': ['sqrt', 'log2']
    }
    rfc_random = RandomizedSearchCV(base_model, rfc_params, n_iter=30, cv=5, scoring='f1', n_jobs=-1, random_state=42)
    rfc_random.fit(X, y)
    return rfc_random.best_estimator_, rfc_random.best_params_
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    results = {}
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "model": model
        }
    return results