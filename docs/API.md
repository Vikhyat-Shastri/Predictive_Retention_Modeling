# API Documentation

## Customer Churn Prediction REST API

Base URL: `http://localhost:8000`

### Authentication
Currently no authentication required (add JWT/OAuth for production)

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check if API is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-27T10:30:00",
  "models_loaded": {
    "pipeline": true,
    "segmenter": true,
    "explainer": true
  },
  "version": "1.0.0"
}
```

---

### 2. Model Information

**GET** `/model/info`

Get information about the loaded model.

**Response:**
```json
{
  "model_name": "Customer Churn Prediction Model",
  "model_type": "lgbm",
  "version": "1.0.0",
  "features": ["gender", "SeniorCitizen", ...],
  "resampling_method": "smoteenn",
  "last_trained": "2025-10-28",
  "performance_metrics": {
    "accuracy": 0.740,
    "precision": 0.507,
    "recall": 0.773,
    "f1_score": 0.612,
    "roc_auc": 0.822
  }
}
```

---

### 3. Single Prediction

**POST** `/predict`

Predict churn for a single customer.

**Request Body:**
```json
{
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
  "TotalCharges": 840.00
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "churn_probability": 0.78,
  "retention_probability": 0.22,
  "risk_level": "High",
  "segment": 2,
  "timestamp": "2025-10-27T10:30:00",
  "recommendations": [
    "URGENT: Implement immediate retention campaign",
    "Offer substantial discount or service upgrade",
    "Schedule priority customer service call"
  ]
}
```

---

### 4. Batch Prediction

**POST** `/predict/batch`

Predict churn for multiple customers.

**Request Body:**
```json
[
  {
    "gender": "Female",
    "SeniorCitizen": 0,
    ...
  },
  {
    "gender": "Male",
    "SeniorCitizen": 1,
    ...
  }
]
```

**Response:**
```json
{
  "total_predictions": 2,
  "high_risk_count": 1,
  "medium_risk_count": 0,
  "low_risk_count": 1,
  "processing_time": 0.15,
  "predictions": [...]
}
```

---

### 5. File Upload Prediction

**POST** `/predict/file`

Upload CSV file for batch predictions.

**Request:**
- Content-Type: `multipart/form-data`
- File: CSV with customer data

**Response:**
- CSV file with predictions

---

### 6. Explain Prediction

**POST** `/explain`

Get SHAP explanation for a prediction.

**Request Body:** (Same as `/predict`)

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "churn_probability": 0.78,
  "explanation": "SHAP feature contributions",
  "top_features": {
    "Contract": "Month-to-month contracts increase churn risk",
    "tenure": "Lower tenure correlates with higher churn"
  }
}
```

---

### 7. Segment Information

**GET** `/segments/info`

Get customer segmentation information.

**Response:**
```json
{
  "n_clusters": 4,
  "segments": [
    {
      "Segment": "Segment 0",
      "Size": 1200,
      "Size_Pct": 17.0,
      "Churn_Rate": 45.2,
      "Avg_Tenure": 8.5,
      "Avg_MonthlyCharges": 75.30
    },
    ...
  ]
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid input data"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Prediction failed: <error message>"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded"
}
```

---

## Rate Limiting

Currently no rate limiting (implement for production)

---

## Examples

### Python

```python
import requests

# Make prediction
response = requests.post('http://localhost:8000/predict', json={
    "gender": "Female",
    "SeniorCitizen": 0,
    "tenure": 12,
    # ... other fields
})

result = response.json()
print(f"Churn Risk: {result['churn_probability']:.2%}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "tenure": 12,
    ...
  }'
```

### JavaScript

```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    gender: 'Female',
    SeniorCitizen: 0,
    tenure: 12,
    // ... other fields
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

Visit `http://localhost:8000/redoc` for ReDoc documentation.
