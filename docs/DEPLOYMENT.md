# Deployment Guide

## Deployment Options

### Option 1: Local Development

#### Prerequisites
- Python 3.10+
- pip
- Virtual environment (recommended)

#### Steps
1. Clone repository
2. Create virtual environment
3. Install dependencies
4. Train models
5. Run applications

```bash
git clone <repo-url>
cd Predictive_Retention_Modeling
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_models.py
streamlit run app/streamlit_app.py
```

---

### Option 2: Docker Containers

#### Prerequisites
- Docker
- Docker Compose

#### Steps
1. Build images
2. Start services
3. Access applications

```bash
# Build and start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Services:**
- Streamlit: http://localhost:8501
- API: http://localhost:8000

---

### Option 3: Cloud Deployment (AWS)

#### Using AWS Elastic Beanstalk

1. **Install EB CLI**
```bash
pip install awsebcli
```

2. **Initialize**
```bash
eb init -p python-3.10 churn-prediction
```

3. **Create environment**
```bash
eb create churn-prediction-env
```

4. **Deploy**
```bash
eb deploy
```

#### Using AWS ECS (Docker)

1. **Push to ECR**
```bash
aws ecr create-repository --repository-name churn-prediction-api
docker tag churn-prediction-api:latest <account>.dkr.ecr.<region>.amazonaws.com/churn-prediction-api:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/churn-prediction-api:latest
```

2. **Create ECS Task Definition**
3. **Create ECS Service**
4. **Configure Load Balancer**

---

### Option 4: Cloud Deployment (Google Cloud)

#### Using Cloud Run

1. **Build and push**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/churn-prediction-api
```

2. **Deploy**
```bash
gcloud run deploy churn-prediction-api \
  --image gcr.io/PROJECT_ID/churn-prediction-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

### Option 5: Cloud Deployment (Azure)

#### Using Azure Container Instances

1. **Login**
```bash
az login
```

2. **Create container group**
```bash
az container create \
  --resource-group myResourceGroup \
  --name churn-prediction \
  --image churn-prediction-api:latest \
  --dns-name-label churn-prediction \
  --ports 8000
```

---

## Environment Variables

Create `.env` file:

```env
# Model Configuration
MODEL_PATH=models/churn_model_pipeline.pkl
SEGMENTATION_MODEL_PATH=models/segmentation_model.pkl
EXPLAINER_PATH=models/shap_explainer.pkl

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log

# Database (if applicable)
# DATABASE_URL=postgresql://user:pass@localhost:5432/churn

# Security
# SECRET_KEY=your-secret-key
# JWT_ALGORITHM=HS256
```

---

## Production Checklist

### Security
- [ ] Add authentication (JWT/OAuth)
- [ ] Enable HTTPS/TLS
- [ ] Implement rate limiting
- [ ] Add CORS configuration
- [ ] Sanitize inputs
- [ ] Environment variable security

### Performance
- [ ] Enable caching
- [ ] Add load balancing
- [ ] Configure auto-scaling
- [ ] Optimize model loading
- [ ] Add CDN for static assets

### Monitoring
- [ ] Setup logging
- [ ] Add metrics collection
- [ ] Configure alerts
- [ ] Performance monitoring
- [ ] Error tracking (Sentry)

### Reliability
- [ ] Health checks
- [ ] Graceful shutdown
- [ ] Database backups
- [ ] Model versioning
- [ ] Rollback strategy

### Documentation
- [ ] API documentation
- [ ] User guide
- [ ] Deployment guide
- [ ] Troubleshooting guide

---

## Scaling Considerations

### Horizontal Scaling
- Deploy multiple API instances
- Use load balancer (Nginx, AWS ALB)
- Redis for session management
- Database connection pooling

### Vertical Scaling
- Increase container resources
- Optimize model inference
- Use GPU for predictions (if applicable)

### Model Serving
- Consider TensorFlow Serving
- Use model registry (MLflow)
- Implement A/B testing
- Canary deployments

---

## Monitoring & Logging

### Application Metrics
- Request count
- Response time
- Error rate
- Prediction latency

### Model Metrics
- Prediction distribution
- Feature drift
- Model performance degradation
- Retraining triggers

### Tools
- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **ELK Stack** - Log aggregation
- **Datadog** - APM
- **New Relic** - Full stack monitoring

---

## Troubleshooting

### Models Not Loading
```bash
# Check if model files exist
ls -la models/

# Verify file permissions
chmod 644 models/*.pkl

# Test model loading
python -c "from src.pipeline import ChurnPredictionPipeline; ChurnPredictionPipeline.load_model('models/churn_model_pipeline.pkl')"
```

### Container Issues
```bash
# Check logs
docker-compose logs api
docker-compose logs streamlit

# Rebuild containers
docker-compose build --no-cache

# Check network
docker network ls
```

### Performance Issues
- Check resource usage: `docker stats`
- Optimize batch size
- Add caching
- Profile code
- Use lighter models

---

## Backup & Recovery

### Model Backups
```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Restore models
tar -xzf models_backup_20251027.tar.gz
```

### Database Backups (if applicable)
```bash
# PostgreSQL
pg_dump dbname > backup.sql

# Restore
psql dbname < backup.sql
```

---

## CI/CD Pipeline

The project includes GitHub Actions workflow:

1. **Test** - Run unit tests
2. **Lint** - Code quality checks
3. **Build** - Docker images
4. **Deploy** - To cloud platform

Customize `.github/workflows/ci-cd.yml` for your needs.

---

## Cost Optimization

### Cloud Costs
- Use spot instances
- Auto-scaling policies
- Reserved instances for production
- Cost monitoring and alerts

### Infrastructure
- Right-size containers
- Optimize storage
- Use caching effectively
- Monitor unused resources

---

## Support & Maintenance

### Regular Tasks
- Monitor model performance
- Update dependencies
- Security patches
- Retrain models periodically
- Review logs

### Version Updates
1. Test in staging
2. Backup current version
3. Deploy new version
4. Monitor for issues
5. Rollback if needed

---

## Contact

For deployment issues or questions:
- GitHub Issues: [Project Issues](https://github.com/Vikhyat-Shastri/Predictive_Retention_Modeling/issues)
- Documentation: [Project Wiki](https://github.com/Vikhyat-Shastri/Predictive_Retention_Modeling/wiki)
