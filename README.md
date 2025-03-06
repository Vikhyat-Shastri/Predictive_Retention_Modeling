# Customer Churn Prediction Model

## Overview
This project focuses on predicting customer churn using machine learning models. The primary objective is to classify whether a customer is likely to churn based on various features such as monthly charges, contract duration, online security, and billing preferences. The dataset used contains 7,000+ customer records with a churn ratio of 27:73.
Customer churn prediction is critical for businesses looking to retain customers and optimize revenue. By identifying at-risk customers early, businesses can take proactive measures such as offering personalized promotions or improved service plans. This project demonstrates how machine learning techniques can be leveraged to analyze customer behavior and predict churn with high accuracy.

## Features
- Data preprocessing including handling missing values, encoding categorical variables, and feature scaling.
- Addressing class imbalance using SMOTEENN and ADASYN to improve recall and prediction stability.
- Implementation of multiple machine learning models:
  - Decision Tree
  - Random Forest
  - XGBoost
- Performance evaluation using accuracy, precision, recall, and F1-score.
- Model optimization for different business objectives:
  - High-recall model (73%) to minimize false negatives and prevent customer loss.
  - High-accuracy model (75.7%) to reduce false positives and optimize retention efforts.

## Dataset
The dataset consists of 7,000+ customer records with various demographic and service-related features. Churn labels indicate whether a customer has left the service.

## Installation
To run this project, install the following dependencies:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn xgboost
```

## Usage
Run the Jupyter Notebook to execute the churn prediction pipeline:

```bash
jupyter notebook churn_analysis.ipynb
```

## Results
- The Random Forest model achieved the highest accuracy of 75.7%.
- Exploratory Data Analysis (EDA) identified key churn factors: higher monthly charges, shorter contract duration, lack of online security, paperless billing, and senior citizen status.

## Future Improvements
- Implementing deep learning models such as LSTMs or transformers for improved prediction.
- Developing a real-time API to integrate the model with business applications.
- Enhancing feature engineering by incorporating additional customer interaction data.

## Contributions
Contributions are welcome! If you have any improvements or suggestions, feel free to fork this repository and submit a pull request.

