
# Predictive Retention Modeling

This repository provides a complete workflow for predicting customer churn using machine learning. The process includes:
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Model training and hyperparameter tuning
- Inference and model persistence

## Folder Structure

- `data/`: Raw dataset (`Telco_customer_churn.csv`)
- `models/`: Trained model files (`churn_model.pkl`)
- `notebooks/`: Jupyter notebooks for analysis
- `src/`: Python scripts for preprocessing, EDA, training, inference, and utilities

## Setup

Install all required dependencies:
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install pandas scikit-learn matplotlib seaborn xgboost imbalanced-learn
```

## Usage

1. **Preprocessing**: Clean and encode the data
    ```python
    from src.preprocess import load_and_clean
    df = load_and_clean('data/Telco_customer_churn.csv')
    ```
2. **EDA**: Visualize features and correlations
    ```python
    from src.eda import plot_categorical_churn, plot_numerical_churn, plot_correlation
    plot_categorical_churn(df)
    plot_numerical_churn(df)
    plot_correlation(df)
    ```
3. **Training**: Train models and tune hyperparameters
    ```python
    from src.train import split_and_resample, train_models, tune_xgboost, tune_rf
    X_train, X_test, y_train, y_test, X_resampled, y_resampled, _, _ = split_and_resample(df)
    results = train_models(X_resampled, y_resampled)
    # Hyperparameter tuning example:
    # best_xgb, params = tune_xgboost(X_resampled, y_resampled, XGBClassifier())
    ```
4. **Inference**: Load models and predict
    ```python
    from src.inference import load_model, predict
    model, features = load_model('models/churn_model.pkl')
    # predictions = predict(model, X_test)
    ```
5. **Utilities**: Save and load models
    ```python
    from src.utils import save_model, load_model
    save_model(model, X_train.columns.tolist(), 'models/churn_model.pkl')
    model, features = load_model('models/churn_model.pkl')
    ```

## Requirements

- pandas
- scikit-learn
- matplotlib
- seaborn
- xgboost
- imbalanced-learn

## Notes

- Ensure all dependencies are installed before running scripts.
- For visualization, use the plotting functions in `src/eda.py`.
- For hyperparameter tuning, see `src/train.py` for examples using RandomizedSearchCV.

## Results
- The Random Forest model achieved the highest accuracy of 75.7%.
- EDA identified key churn factors: higher monthly charges, shorter contract duration, lack of online security, paperless billing, and senior citizen status.

## Future Improvements
- Implement deep learning models (LSTM, transformers) for improved prediction.
- Develop a real-time API for business integration.
- Enhance feature engineering with additional customer data.

## License

MIT

