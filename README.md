
# 🎯 Customer Churn Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **An end-to-end production-ready machine learning system for predicting customer churn with advanced explainability and interactive deployment**

## 🌟 Project Highlights

This is a **comprehensive, production-ready** customer churn prediction system that goes beyond basic ML modeling. Built with industry best practices, it demonstrates:

- ✅ **Advanced ML Pipeline** - Production-grade sklearn Pipeline with automated preprocessing
- ✅ **Multiple ML Algorithms** - Random Forest, XGBoost, LightGBM with ensemble methods
- ✅ **Class Imbalance Handling** - SMOTEENN and ADASYN implementation
- ✅ **Model Explainability** - SHAP integration for transparent AI decisions
- ✅ **Customer Segmentation** - K-Means clustering with business insights
- ✅ **Interactive Web App** - Streamlit dashboard for real-time predictions
- ✅ **REST API** - FastAPI backend for production integration
- ✅ **Docker Deployment** - Containerized application ready for cloud deployment
- ✅ **Comprehensive Testing** - Unit tests and API tests with pytest
- ✅ **CI/CD Pipeline** - GitHub Actions for automated testing and deployment

## 📊 Business Impact

- **Predict** customer churn with **74%+ accuracy** (77% recall)
- **Explain** predictions using SHAP values for actionable insights
- **Segment** customers into distinct groups for targeted retention strategies
- **Deploy** easily with Docker containers or cloud platforms
- **Scale** with production-ready API and monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐         │
│  │  Raw Data  │→ │ Preprocessing│→ │Feature Eng.  │         │
│  └────────────┘  └─────────────┘  └──────────────┘         │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   ML Pipeline Layer                          │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐      │
│  │   SMOTE    │→ │ ML Ensemble  │→ │ SHAP Explainer │      │
│  │  ADASYN    │  │ RF/XGB/LGBM  │  │   K-Means      │      │
│  └────────────┘  └──────────────┘  └────────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                 Application Layer                            │
│  ┌─────────────────┐              ┌──────────────────┐      │
│  │  FastAPI REST   │              │   Streamlit      │      │
│  │     API         │              │    Web App       │      │
│  │  (Port 8000)    │              │  (Port 8501)     │      │
│  └─────────────────┘              └──────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Predictive_Retention_Modeling/
├── 📊 data/
│   └── Telco_customer_churn.csv        # Dataset
│
├── 🎓 models/                          # Trained models & artifacts
│   ├── churn_model_pipeline.pkl        # Main prediction model
│   ├── segmentation_model.pkl          # Customer segmentation
│   ├── shap_explainer.pkl              # SHAP explainer
│   ├── segments_visualization.png      # Segment analysis
│   ├── shap_summary.png                # SHAP summary plots
│   └── explanations/                   # Individual explanations
│
├── 📓 notebooks/
│   └── churn_analysis.ipynb            # Exploratory analysis
│
├── 🧠 src/                             # Core ML modules
│   ├── pipeline.py                     # Advanced ML pipeline
│   ├── segmentation.py                 # Customer segmentation
│   ├── explainability.py               # SHAP integration
│   ├── preprocess.py                   # Data preprocessing
│   ├── train.py                        # Model training
│   ├── eda.py                          # Data analysis
│   ├── inference.py                    # Predictions
│   └── utils.py                        # Utilities
│
├── 🌐 app/                             # Web applications
│   ├── streamlit_app.py                # Interactive dashboard
│   └── api.py                          # FastAPI REST API
│
├── 🧪 tests/                           # Test suite
│   ├── test_pipeline.py                # Pipeline tests
│   └── test_api.py                     # API tests
│
├── 🐳 Docker files
│   ├── Dockerfile.api                  # API container
│   ├── Dockerfile.streamlit            # Streamlit container
│   └── docker-compose.yml              # Orchestration
│
├── ⚙️ .github/workflows/
│   └── ci-cd.yml                       # CI/CD pipeline
│
├── 📜 Configuration files
│   ├── requirements.txt                # Python dependencies
│   ├── setup.py                        # Package setup
│   └── train_models.py                 # Training script
│
└── 📖 README.md                        # This file
```

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/Vikhyat-Shastri/Predictive_Retention_Modeling.git
cd Predictive_Retention_Modeling

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Train Models

```bash
# Train all models (Pipeline, Segmentation, SHAP)
python train_models.py
```

This will:
- ✅ Load and preprocess data
- ✅ Train multiple ML models (RF, XGBoost, LightGBM, Ensembles)
- ✅ Perform customer segmentation
- ✅ Generate SHAP explanations
- ✅ Save all models and artifacts

### 3️⃣ Run Applications

#### Option A: Streamlit Web App

```bash
streamlit run app/streamlit_app.py
```

Visit: `http://localhost:8501`

#### Option B: FastAPI REST API

```bash
cd app
python api.py
```

Visit API docs: `http://localhost:8000/docs`

#### Option C: Docker (Recommended for Production)

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- 🌐 Streamlit: `http://localhost:8501`
- 🔌 API: `http://localhost:8000`
- 📚 API Docs: `http://localhost:8000/docs`

## 🎯 Features

### 1. 🔮 Real-Time Churn Prediction

- **Input customer data** through web form or API
- **Get instant predictions** with probability scores
- **Risk assessment** (Low/Medium/High)
- **Actionable recommendations** based on risk level

### 2. 🔍 Model Explainability (SHAP)

- **Feature importance** - See which factors matter most
- **Individual explanations** - Understand each prediction
- **Force plots** - Visualize feature contributions
- **Waterfall plots** - Step-by-step prediction breakdown

### 3. 👥 Customer Segmentation

- **K-Means clustering** identifies 4 distinct customer groups
- **Segment profiles** with demographics and behavior
- **Churn analysis** by segment
- **Targeted strategies** for each segment

### 4. 📊 Interactive Dashboard

Multi-page Streamlit application:
- 🏠 **Home** - Overview and introduction
- 🔮 **Prediction** - Make churn predictions
- 🔍 **Explanation** - SHAP-based insights
- 👥 **Segmentation** - Customer group analysis
- 📈 **Performance** - Model metrics

### 5. 🔌 Production REST API

FastAPI endpoints:
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `POST /predict/file` - Upload CSV for predictions
- `POST /explain` - Get SHAP explanation
- `GET /segments/info` - Segment information

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **LightGBM (Current)** | 74.0% | 50.7% | 77.3% | 61.2% | 82.2% |

**Note:** Current model is optimized for high recall (77.3%) to catch more potential churners, accepting lower precision. This is ideal for retention campaigns where missing a churner is more costly than false alarms.

## 🛠️ Technology Stack

### Machine Learning
- **Scikit-learn** - ML pipeline and preprocessing
- **LightGBM** - Gradient boosting
- **XGBoost** - Gradient boosting
- **Imbalanced-learn** - SMOTE, ADASYN

### Explainability & Analysis
- **SHAP** - Model explanations
- **K-Means** - Customer segmentation
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Visualization
- **Plotly** - Interactive plots
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical plots

### Web & API
- **Streamlit** - Interactive dashboard
- **FastAPI** - REST API
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### DevOps & Testing
- **Docker** - Containerization
- **Pytest** - Testing framework
- **GitHub Actions** - CI/CD
- **Black** - Code formatting

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v

# Run API tests
pytest tests/test_api.py -v
```

## 📈 Usage Examples

### Python API

```python
from src.pipeline import ChurnPredictionPipeline
from src.explainability import ModelExplainer
import pandas as pd

# Load model
pipeline = ChurnPredictionPipeline.load_model('models/churn_model_pipeline.pkl')

# Make prediction
customer_data = pd.DataFrame({...})  # Your customer data
prediction = pipeline.predict(customer_data)
probability = pipeline.predict_proba(customer_data)

# Get explanation
explainer = ModelExplainer(pipeline.pipeline)
explanation = explainer.explain_instance(customer_data, index=0)
```

### REST API

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
print(f"Prediction: {result['prediction_label']}")
print(f"Churn Risk: {result['churn_probability']:.2%}")
```

## 🎓 Key Learnings & Innovations

1. **Production-Ready Pipeline** - Full sklearn Pipeline with custom transformers
2. **Advanced Imbalance Handling** - Compared SMOTEENN vs ADASYN
3. **Ensemble Methods** - Voting and stacking classifiers
4. **SHAP Integration** - Transparent AI with feature explanations
5. **Customer Segmentation** - K-Means with business insights
6. **API Design** - RESTful API with proper validation
7. **Containerization** - Docker multi-service deployment
8. **Testing** - Comprehensive unit and integration tests
9. **CI/CD** - Automated testing and deployment pipeline

## 👥 Team & Contributions

This project demonstrates proficiency in:
- **Machine Learning Engineering** - Pipeline design, model training
- **Data Science** - EDA, segmentation, feature engineering
- **MLOps** - Deployment, monitoring, CI/CD
- **Software Engineering** - API design, testing, documentation
- **DevOps** - Docker, containerization, orchestration

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- Dataset: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- SHAP Library: [shap.readthedocs.io](https://shap.readthedocs.io)

## 📧 Contact

**Vikhyat Shastri**
- GitHub: [@Vikhyat-Shastri](https://github.com/Vikhyat-Shastri)
- Project: [Predictive_Retention_Modeling](https://github.com/Vikhyat-Shastri/Predictive_Retention_Modeling)

---

⭐ **Star this repository if you find it helpful!** ⭐