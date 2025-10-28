import matplotlib.pyplot as plt
import seaborn as sns


def plot_categorical_churn(df):
    for predictor in df.drop(columns=["Churn", "TotalCharges", "MonthlyCharges"]):
        plt.figure(figsize=(4, 2))
        sns.countplot(data=df, x=predictor, hue="Churn")
        plt.show()


def plot_numerical_churn(df):
    sns.lmplot(
        data=df,
        x="MonthlyCharges",
        y="TotalCharges",
        fit_reg=False,
        height=4,
        aspect=1.7,
    )
    plt.show()
    Mth = sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 0)], color="Red", shade=True)
    Mth = sns.kdeplot(
        df.MonthlyCharges[(df["Churn"] == 1)], ax=Mth, color="Blue", shade=True
    )
    Mth.legend(["No Churn", "Churn"], loc="upper right")
    Mth.set_ylabel("Density")
    Mth.set_xlabel("Monthly Charges")
    Mth.set_title("Monthly charges by churn")
    plt.show()
    Mth = sns.kdeplot(df.TotalCharges[(df["Churn"] == 0)], color="Red", shade=True)
    Mth = sns.kdeplot(
        df.TotalCharges[(df["Churn"] == 1)], ax=Mth, color="Blue", shade=True
    )
    Mth.legend(["No Churn", "Churn"], loc="upper right")
    Mth.set_ylabel("Density")
    Mth.set_xlabel("Total Charges")
    Mth.set_title("Total charges by churn")
    plt.show()


def plot_correlation(df):
    plt.figure(figsize=(20, 8))
    df.corr()["Churn"].sort_values(ascending=False).plot(kind="bar")
    plt.show()
    plt.figure(figsize=(12, 12))
    sns.heatmap(df.corr(), cmap="Paired")
    plt.show()


def plot_churn_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="Churn")
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.show()


def plot_numerical_features(df):
    numerical_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    df[numerical_features].hist(
        bins=15, figsize=(15, 6), layout=(2, len(numerical_features) // 2)
    )
    plt.tight_layout()
    plt.show()
