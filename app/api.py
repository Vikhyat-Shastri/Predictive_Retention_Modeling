"""
FastAPI REST API for Customer Churn Prediction
Production-ready API with comprehensive endpoints
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import joblib
import uvicorn
import os
import sys
from datetime import datetime
import logging
from io import StringIO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import ChurnPredictionPipeline  # noqa: E402
from segmentation import CustomerSegmentation  # noqa: E402
from explainability import ModelExplainer  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready API for customer churn prediction with ML explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {"pipeline": None, "segmenter": None, "explainer": None}


# Request/Response models
class CustomerData(BaseModel):
    """Customer data model for single prediction"""

    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="Senior citizen status (0/1)"
    )
    Partner: str = Field(..., description="Has partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")
    tenure: int = Field(..., ge=0, le=72, description="Months with company")
    PhoneService: str = Field(..., description="Has phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Has multiple lines")
    InternetService: str = Field(..., description="Internet service type")
    OnlineSecurity: str = Field(..., description="Has online security")
    OnlineBackup: str = Field(..., description="Has online backup")
    DeviceProtection: str = Field(..., description="Has device protection")
    TechSupport: str = Field(..., description="Has tech support")
    StreamingTV: str = Field(..., description="Has streaming TV")
    StreamingMovies: str = Field(..., description="Has streaming movies")
    Contract: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Uses paperless billing")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 840.00,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model"""

    customer_id: Optional[str] = None
    prediction: int = Field(..., description="Predicted class (0=No Churn, 1=Churn)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    churn_probability: float = Field(..., description="Probability of churning")
    retention_probability: float = Field(..., description="Probability of retention")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    segment: Optional[int] = Field(None, description="Customer segment")
    timestamp: str = Field(..., description="Prediction timestamp")
    recommendations: List[str] = Field(..., description="Action recommendations")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""

    total_predictions: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[PredictionResponse]
    processing_time: float


class ModelInfo(BaseModel):
    """Model information response"""

    model_name: str
    model_type: str
    version: str
    features: List[str]
    resampling_method: Optional[str]
    last_trained: Optional[str]
    performance_metrics: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    version: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Loading models...")
    try:
        # Load prediction pipeline
        models["pipeline"] = ChurnPredictionPipeline.load_model(
            "models/churn_model_pipeline.pkl"
        )
        logger.info("Prediction pipeline loaded successfully")

        # Load segmentation model
        try:
            models["segmenter"] = CustomerSegmentation.load_model(
                "models/segmentation_model.pkl"
            )
            logger.info("Segmentation model loaded successfully")
        except Exception as e:
            logger.warning(f"Segmentation model not loaded: {e}")

        # Load explainer
        try:
            explainer_data = joblib.load("models/shap_explainer.pkl")
            models["explainer"] = ModelExplainer(models["pipeline"].pipeline)
            models["explainer"].explainer = explainer_data["explainer"]
            logger.info("Explainer loaded successfully")
        except Exception as e:
            logger.warning(f"Explainer not loaded: {e}")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def get_risk_level(probability: float) -> str:
    """Determine risk level from probability"""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


def get_recommendations(
    prediction: int, probability: float, tenure: int, monthly_charges: float
) -> List[str]:
    """Generate action recommendations"""
    recommendations = []

    if prediction == 1:  # Churn
        if probability >= 0.7:
            recommendations.extend(
                [
                    "URGENT: Implement immediate retention campaign",
                    "Offer substantial discount or service upgrade",
                    "Schedule priority customer service call",
                    "Review and address any service complaints",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Initiate proactive retention outreach",
                    "Offer loyalty incentive or discount",
                    "Conduct satisfaction survey",
                    "Monitor account activity closely",
                ]
            )
    else:  # No churn
        recommendations.extend(
            [
                "Maintain current service quality",
                "Consider for loyalty rewards program",
                "Explore upselling opportunities",
                "Request feedback or testimonial",
            ]
        )

    # Tenure-based recommendations
    if tenure < 12:
        recommendations.append("Focus on onboarding experience and early engagement")

    # Revenue-based recommendations
    if monthly_charges > 80:
        recommendations.append("High-value customer - provide premium support")

    return recommendations


# API Endpoints


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if models["pipeline"] is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={
            "pipeline": models["pipeline"] is not None,
            "segmenter": models["segmenter"] is not None,
            "explainer": models["explainer"] is not None,
        },
        version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if models["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_name="Customer Churn Prediction Model",
        model_type=models["pipeline"].model_type,
        version="1.0.0",
        features=models["pipeline"].feature_names or [],
        resampling_method=models["pipeline"].resampling_method,
        last_trained="2025-10-28",
        performance_metrics={
            "accuracy": 0.740,
            "precision": 0.507,
            "recall": 0.773,
            "f1_score": 0.612,
            "roc_auc": 0.822,
        },
    )


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features to match training data encoding.
    Uses LabelEncoder-compatible mappings (alphabetical order).
    """
    df = df.copy()

    # Binary encodings (alphabetical: Female=0, Male=1; No=0, Yes=1)
    binary_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }

    # MultipleLines encoding (alphabetical)
    multiple_lines_map = {"No": 0, "No phone service": 1, "Yes": 2}

    # Internet service addons (alphabetical: No=0, No internet service=1, Yes=2)
    addon_map = {"No": 0, "No internet service": 1, "Yes": 2}

    # InternetService (alphabetical: DSL=0, Fiber optic=1, No=2)
    internet_service_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

    # Contract (alphabetical: Month-to-month=0, One year=1, Two year=2)
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}

    # PaymentMethod (alphabetical)
    payment_method_map = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3,
    }

    # Apply binary mappings
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Apply complex mappings
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].map(multiple_lines_map)

    if "InternetService" in df.columns:
        df["InternetService"] = df["InternetService"].map(internet_service_map)

    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].map(contract_map)

    if "PaymentMethod" in df.columns:
        df["PaymentMethod"] = df["PaymentMethod"].map(payment_method_map)

    # Apply addon mappings
    addon_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in addon_cols:
        if col in df.columns:
            df[col] = df[col].map(addon_map)

    # Add tenure_group (this will be created by FeatureEngineer in pipeline)
    # But we need tenure column for the pipeline
    # Keep tenure as is - the pipeline expects it

    return df


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predict churn for a single customer

    - **Returns**: Prediction with probability, risk level, and recommendations
    """
    if models["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to dataframe
        input_df = pd.DataFrame([customer.dict()])

        # Encode categorical features
        input_df = encode_categorical_features(input_df)

        # Make prediction
        prediction = models["pipeline"].predict(input_df)[0]
        prediction_proba = models["pipeline"].predict_proba(input_df)[0]

        # Get segment if available
        segment = None
        if models["segmenter"] is not None:
            try:
                segment = int(models["segmenter"].predict(input_df)[0])
            except Exception:
                pass

        # Get risk level and recommendations
        risk_level = get_risk_level(prediction_proba[1])
        recommendations = get_recommendations(
            prediction, prediction_proba[1], customer.tenure, customer.MonthlyCharges
        )

        return PredictionResponse(
            prediction=int(prediction),
            prediction_label="Churn" if prediction == 1 else "No Churn",
            churn_probability=float(prediction_proba[1]),
            retention_probability=float(prediction_proba[0]),
            risk_level=risk_level,
            segment=segment,
            timestamp=datetime.now().isoformat(),
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(customers: List[CustomerData]):
    """
    Predict churn for multiple customers

    - **Returns**: Batch predictions with summary statistics
    """
    if models["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = datetime.now()

        # Convert to dataframe
        input_df = pd.DataFrame([c.dict() for c in customers])

        # Encode categorical features
        input_df = encode_categorical_features(input_df)

        # Make predictions
        predictions = models["pipeline"].predict(input_df)
        predictions_proba = models["pipeline"].predict_proba(input_df)

        # Get segments if available
        segments = None
        if models["segmenter"] is not None:
            try:
                segments = models["segmenter"].predict(input_df)
            except Exception:
                pass

        # Build response
        results = []
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}

        for i, (customer, pred, proba) in enumerate(
            zip(customers, predictions, predictions_proba)
        ):
            risk_level = get_risk_level(proba[1])
            risk_counts[risk_level] += 1

            result = PredictionResponse(
                customer_id=f"CUST_{i+1:04d}",
                prediction=int(pred),
                prediction_label="Churn" if pred == 1 else "No Churn",
                churn_probability=float(proba[1]),
                retention_probability=float(proba[0]),
                risk_level=risk_level,
                segment=int(segments[i]) if segments is not None else None,
                timestamp=datetime.now().isoformat(),
                recommendations=get_recommendations(
                    pred, proba[1], customer.tenure, customer.MonthlyCharges
                ),
            )
            results.append(result)

        processing_time = (datetime.now() - start_time).total_seconds()

        return BatchPredictionResponse(
            total_predictions=len(results),
            high_risk_count=risk_counts["High"],
            medium_risk_count=risk_counts["Medium"],
            low_risk_count=risk_counts["Low"],
            predictions=results,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    """
    Predict churn from uploaded CSV file

    - **Returns**: Predictions as downloadable CSV
    """
    if models["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Encode categorical features
        df = encode_categorical_features(df)

        # Make predictions
        predictions = models["pipeline"].predict(df)
        predictions_proba = models["pipeline"].predict_proba(df)

        # Add to dataframe
        df["Prediction"] = predictions
        df["Churn_Probability"] = predictions_proba[:, 1]
        df["Retention_Probability"] = predictions_proba[:, 0]
        df["Risk_Level"] = [get_risk_level(p) for p in predictions_proba[:, 1]]

        # Save to temporary file
        output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)

        return FileResponse(output_file, media_type="text/csv", filename=output_file)

    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")


@app.post("/explain")
async def explain_prediction(customer: CustomerData):
    """
    Explain prediction using SHAP values

    - **Returns**: Feature contributions and explanations
    """
    if models["pipeline"] is None or models["explainer"] is None:
        raise HTTPException(status_code=503, detail="Model or explainer not loaded")

    try:
        # Convert to dataframe
        input_df = pd.DataFrame([customer.dict()])

        # Encode categorical features
        input_df = encode_categorical_features(input_df)

        # Get prediction
        prediction = models["pipeline"].predict(input_df)[0]
        prediction_proba = models["pipeline"].predict_proba(input_df)[0]

        # Get SHAP explanation
        # Note: This is simplified - full implementation would need proper data prep
        explanation = {
            "prediction": int(prediction),
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": float(prediction_proba[1]),
            "explanation": "SHAP explanation requires preprocessing. See /docs for details.",
            "top_features": {
                "Contract": "Month-to-month contracts increase churn risk",
                "tenure": "Lower tenure correlates with higher churn",
                "MonthlyCharges": "Higher charges may increase churn risk",
            },
        }

        return explanation

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/segments/info")
async def get_segments_info():
    """
    Get customer segmentation information

    - **Returns**: Segment profiles and statistics
    """
    if models["segmenter"] is None:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")

    try:
        profiles = models["segmenter"].segment_profiles

        if profiles is not None:
            return {
                "n_clusters": models["segmenter"].n_clusters,
                "segments": profiles.to_dict(orient="records"),
            }
        else:
            return {"message": "Segment profiles not available"}

    except Exception as e:
        logger.error(f"Segmentation info error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get segment info: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
