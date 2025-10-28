"""
Customer Segmentation using K-Means Clustering
Identify distinct customer groups and analyze their churn patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerSegmentation:
    """
    K-Means based customer segmentation with comprehensive analysis
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """
        Initialize segmentation

        Args:
            n_clusters: Number of customer segments
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        self.feature_names = None
        self.segment_profiles = None

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for clustering

        Args:
            df: Input dataframe

        Returns:
            Feature dataframe for clustering
        """
        # Select relevant features for segmentation
        clustering_features = df.copy()

        # Keep only numerical features for clustering
        numerical_cols = clustering_features.select_dtypes(
            include=["int64", "float64"]
        ).columns
        clustering_features = clustering_features[numerical_cols]

        # Remove Churn if present (we'll use it for analysis but not clustering)
        if "Churn" in clustering_features.columns:
            clustering_features = clustering_features.drop("Churn", axis=1)

        self.feature_names = clustering_features.columns.tolist()
        logger.info(f"Using {len(self.feature_names)} features for clustering")

        return clustering_features

    def find_optimal_clusters(
        self, X: pd.DataFrame, max_clusters: int = 10
    ) -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method and silhouette score

        Args:
            X: Feature dataframe
            max_clusters: Maximum number of clusters to try

        Returns:
            Dictionary with metrics for each k
        """
        X_scaled = self.scaler.fit_transform(X)

        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        k_range = range(2, max_clusters + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))

            logger.info(
                f"K={k}: Inertia={kmeans.inertia_:.2f}, "
                f"Silhouette={silhouette_scores[-1]:.3f}, "
                f"Davies-Bouldin={davies_bouldin_scores[-1]:.3f}"
            )

        # Plot metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Elbow plot
        axes[0].plot(k_range, inertias, "bo-")
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Inertia")
        axes[0].set_title("Elbow Method")
        axes[0].grid(True)

        # Silhouette score
        axes[1].plot(k_range, silhouette_scores, "go-")
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Analysis (Higher is Better)")
        axes[1].grid(True)

        # Davies-Bouldin score
        axes[2].plot(k_range, davies_bouldin_scores, "ro-")
        axes[2].set_xlabel("Number of Clusters (k)")
        axes[2].set_ylabel("Davies-Bouldin Score")
        axes[2].set_title("Davies-Bouldin Index (Lower is Better)")
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig("models/cluster_optimization.png", dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "davies_bouldin_scores": davies_bouldin_scores,
        }

    def fit(self, df: pd.DataFrame) -> "CustomerSegmentation":
        """
        Fit K-Means clustering model

        Args:
            df: Input dataframe

        Returns:
            Self
        """
        X = self.prepare_features(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20,
            max_iter=300,
        )

        logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")
        self.kmeans.fit(X_scaled)

        # Fit PCA for visualization
        self.pca.fit(X_scaled)

        logger.info(f"Clustering completed. Inertia: {self.kmeans.inertia_:.2f}")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data

        Args:
            df: Input dataframe

        Returns:
            Array of cluster labels
        """
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

    def create_segment_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create detailed profiles for each segment

        Args:
            df: Original dataframe with all features

        Returns:
            DataFrame with segment profiles
        """
        df_with_segments = df.copy()
        df_with_segments["Segment"] = self.predict(df)

        # Calculate statistics for each segment
        profiles = []

        for segment in range(self.n_clusters):
            segment_data = df_with_segments[df_with_segments["Segment"] == segment]

            profile = {
                "Segment": f"Segment {segment}",
                "Size": len(segment_data),
                "Size_Pct": len(segment_data) / len(df_with_segments) * 100,
                "Churn_Rate": (
                    segment_data["Churn"].mean() * 100
                    if "Churn" in df.columns
                    else None
                ),
                "Avg_Tenure": (
                    segment_data["tenure"].mean() if "tenure" in df.columns else None
                ),
                "Avg_MonthlyCharges": (
                    segment_data["MonthlyCharges"].mean()
                    if "MonthlyCharges" in df.columns
                    else None
                ),
                "Avg_TotalCharges": (
                    segment_data["TotalCharges"].mean()
                    if "TotalCharges" in df.columns
                    else None
                ),
            }

            # Add categorical feature distributions
            categorical_features = [
                "Contract",
                "InternetService",
                "PaymentMethod",
                "PaperlessBilling",
            ]
            for feature in categorical_features:
                if feature in df.columns:
                    mode_value = segment_data[feature].mode()
                    profile[f"Most_Common_{feature}"] = (
                        mode_value[0] if len(mode_value) > 0 else None
                    )

            profiles.append(profile)

        self.segment_profiles = pd.DataFrame(profiles)
        logger.info("\nSegment Profiles:")
        logger.info(self.segment_profiles.to_string())

        return self.segment_profiles

    def visualize_segments(
        self, df: pd.DataFrame, save_path: str = "models/segments_visualization.png"
    ):
        """
        Create comprehensive visualizations of customer segments

        Args:
            df: Original dataframe
            save_path: Path to save the visualization
        """
        df_with_segments = df.copy()
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        df_with_segments["Segment"] = self.kmeans.predict(X_scaled)

        # PCA projection
        X_pca = self.pca.transform(X_scaled)
        df_with_segments["PCA1"] = X_pca[:, 0]
        df_with_segments["PCA2"] = X_pca[:, 1]

        # Create visualization
        plt.figure(figsize=(20, 12))

        # 1. PCA scatter plot
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(
            df_with_segments["PCA1"],
            df_with_segments["PCA2"],
            c=df_with_segments["Segment"],
            cmap="viridis",
            alpha=0.6,
            s=50,
        )
        ax1.set_xlabel("First Principal Component")
        ax1.set_ylabel("Second Principal Component")
        ax1.set_title("Customer Segments (PCA Projection)")
        plt.colorbar(scatter, ax=ax1, label="Segment")

        # 2. Segment sizes
        ax2 = plt.subplot(2, 3, 2)
        segment_counts = df_with_segments["Segment"].value_counts().sort_index()
        ax2.bar(
            segment_counts.index,
            segment_counts.values,
            color="skyblue",
            edgecolor="black",
        )
        ax2.set_xlabel("Segment")
        ax2.set_ylabel("Number of Customers")
        ax2.set_title("Segment Sizes")
        ax2.set_xticks(range(self.n_clusters))

        # 3. Churn rate by segment
        if "Churn" in df.columns:
            ax3 = plt.subplot(2, 3, 3)
            churn_by_segment = df_with_segments.groupby("Segment")["Churn"].mean() * 100
            colors = [
                "red" if x > 30 else "orange" if x > 20 else "green"
                for x in churn_by_segment.values
            ]
            ax3.bar(
                churn_by_segment.index,
                churn_by_segment.values,
                color=colors,
                edgecolor="black",
            )
            ax3.set_xlabel("Segment")
            ax3.set_ylabel("Churn Rate (%)")
            ax3.set_title("Churn Rate by Segment")
            ax3.set_xticks(range(self.n_clusters))
            ax3.axhline(
                y=df["Churn"].mean() * 100,
                color="black",
                linestyle="--",
                label="Overall Average",
            )
            ax3.legend()

        # 4. Monthly charges by segment
        if "MonthlyCharges" in df.columns:
            ax4 = plt.subplot(2, 3, 4)
            df_with_segments.boxplot(column="MonthlyCharges", by="Segment", ax=ax4)
            ax4.set_xlabel("Segment")
            ax4.set_ylabel("Monthly Charges ($)")
            ax4.set_title("Monthly Charges Distribution by Segment")
            plt.suptitle("")  # Remove default title

        # 5. Tenure by segment
        if "tenure" in df.columns:
            ax5 = plt.subplot(2, 3, 5)
            df_with_segments.boxplot(column="tenure", by="Segment", ax=ax5)
            ax5.set_xlabel("Segment")
            ax5.set_ylabel("Tenure (months)")
            ax5.set_title("Tenure Distribution by Segment")
            plt.suptitle("")

        # 6. Segment characteristics heatmap
        ax6 = plt.subplot(2, 3, 6)
        segment_features = df_with_segments.groupby("Segment")[
            self.feature_names
        ].mean()
        # Normalize for better visualization
        segment_features_norm = (segment_features - segment_features.min()) / (
            segment_features.max() - segment_features.min()
        )
        sns.heatmap(
            segment_features_norm.T,
            annot=False,
            cmap="YlOrRd",
            ax=ax6,
            cbar_kws={"label": "Normalized Value"},
        )
        ax6.set_xlabel("Segment")
        ax6.set_ylabel("Features")
        ax6.set_title("Segment Feature Profiles (Normalized)")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Segment visualizations saved to {save_path}")

    def generate_segment_insights(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate business insights for each segment

        Args:
            df: Original dataframe

        Returns:
            Dictionary mapping segment IDs to insight text
        """
        df_with_segments = df.copy()
        df_with_segments["Segment"] = self.predict(df)

        insights = {}

        for segment in range(self.n_clusters):
            segment_data = df_with_segments[df_with_segments["Segment"] == segment]

            # Calculate key metrics
            size = len(segment_data)
            size_pct = size / len(df_with_segments) * 100
            churn_rate = (
                segment_data["Churn"].mean() * 100 if "Churn" in df.columns else 0
            )
            avg_tenure = segment_data["tenure"].mean() if "tenure" in df.columns else 0
            avg_monthly = (
                segment_data["MonthlyCharges"].mean()
                if "MonthlyCharges" in df.columns
                else 0
            )

            # Determine segment characteristics
            if churn_rate > 35:
                risk_level = "HIGH RISK"
            elif churn_rate > 25:
                risk_level = "MODERATE RISK"
            else:
                risk_level = "LOW RISK"

            if avg_tenure < 12:
                tenure_desc = "new customers"
            elif avg_tenure < 36:
                tenure_desc = "mid-term customers"
            else:
                tenure_desc = "long-term loyal customers"

            if avg_monthly < 40:
                value_desc = "low-value"
            elif avg_monthly < 70:
                value_desc = "medium-value"
            else:
                value_desc = "high-value"

            insight = f"""
Segment {segment} - {risk_level}
{'=' * 50}
Size: {size:,} customers ({size_pct:.1f}% of total)
Churn Rate: {churn_rate:.1f}%
Average Tenure: {avg_tenure:.1f} months
Average Monthly Charges: ${avg_monthly:.2f}

Profile: These are {tenure_desc} with {value_desc} service packages.

Recommendations:
"""

            if churn_rate > 30:
                insight += "- Implement urgent retention campaigns\n"
                insight += "- Offer loyalty discounts or service upgrades\n"
                insight += "- Conduct exit surveys to understand pain points\n"
            elif churn_rate > 20:
                insight += "- Monitor closely for early churn signals\n"
                insight += "- Enhance customer engagement programs\n"
                insight += "- Provide proactive customer support\n"
            else:
                insight += "- Maintain current service quality\n"
                insight += "- Explore upselling opportunities\n"
                insight += "- Use as testimonials/case studies\n"

            insights[segment] = insight

        return insights

    def save_model(self, filepath: str):
        """Save the segmentation model"""
        model_data = {
            "kmeans": self.kmeans,
            "scaler": self.scaler,
            "pca": self.pca,
            "feature_names": self.feature_names,
            "n_clusters": self.n_clusters,
            "segment_profiles": self.segment_profiles,
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Segmentation model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved segmentation model"""
        model_data = joblib.load(filepath)

        instance = cls(n_clusters=model_data["n_clusters"])
        instance.kmeans = model_data["kmeans"]
        instance.scaler = model_data["scaler"]
        instance.pca = model_data["pca"]
        instance.feature_names = model_data["feature_names"]
        instance.segment_profiles = model_data["segment_profiles"]

        logger.info(f"Segmentation model loaded from {filepath}")
        return instance


if __name__ == "__main__":
    # Example usage
    from preprocess import load_and_clean

    # Load data
    df = load_and_clean("../data/Telco_customer_churn.csv")

    # Create segmentation
    segmenter = CustomerSegmentation(n_clusters=4)

    # Find optimal clusters
    X = segmenter.prepare_features(df)
    metrics = segmenter.find_optimal_clusters(X, max_clusters=8)

    # Fit the model
    segmenter.fit(df)

    # Create profiles
    profiles = segmenter.create_segment_profiles(df)
    print("\nSegment Profiles:")
    print(profiles)

    # Visualize
    segmenter.visualize_segments(df)

    # Generate insights
    insights = segmenter.generate_segment_insights(df)
    for segment_id, insight in insights.items():
        print(insight)

    # Save model
    segmenter.save_model("../models/segmentation_model.pkl")
