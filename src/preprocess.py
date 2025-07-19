import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(how='any', inplace=True)
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df['tenure_group'] = pd.cut(df.tenure, range(1, 80, 12), right=False, labels=labels)
    df.drop(columns=['customerID', 'tenure'], axis=1, inplace=True)
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
    object_columns = df.select_dtypes(include="object").columns
    for column in object_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
    df['tenure_group'] = df['tenure_group'].astype(str)
    label_encoder = LabelEncoder()
    df['tenure_group'] = label_encoder.fit_transform(df['tenure_group'])
    return df