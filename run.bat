@echo off
REM Setup and Run Script for Customer Churn Prediction System
REM Windows Batch Script

echo ========================================
echo Customer Churn Prediction System
echo Setup and Execution Script
echo ========================================
echo.

:menu
echo.
echo Please select an option:
echo.
echo 1. Setup Environment (First Time)
echo 2. Train Models
echo 3. Run Streamlit Dashboard
echo 4. Run FastAPI Server
echo 5. Run Both (Streamlit + API)
echo 6. Run with Docker
echo 7. Run Tests
echo 8. View Documentation
echo 9. Exit
echo.

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto train
if "%choice%"=="3" goto streamlit
if "%choice%"=="4" goto api
if "%choice%"=="5" goto both
if "%choice%"=="6" goto docker
if "%choice%"=="7" goto tests
if "%choice%"=="8" goto docs
if "%choice%"=="9" goto exit
goto menu

:setup
echo.
echo ========================================
echo Setting Up Environment
echo ========================================
echo.

REM Check Python
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    goto menu
)

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Train models (Option 2)
echo 2. Run applications (Options 3-6)
echo.
pause
goto menu

:train
echo.
echo ========================================
echo Training Models
echo ========================================
echo.

call venv\Scripts\activate
python train_models.py

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
pause
goto menu

:streamlit
echo.
echo ========================================
echo Starting Streamlit Dashboard
echo ========================================
echo.
echo Dashboard will open at: http://localhost:8501
echo Press Ctrl+C to stop
echo.

call venv\Scripts\activate
streamlit run app/streamlit_app.py

pause
goto menu

:api
echo.
echo ========================================
echo Starting FastAPI Server
echo ========================================
echo.
echo API will run at: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Press Ctrl+C to stop
echo.

call venv\Scripts\activate
cd app
python api.py

pause
goto menu

:both
echo.
echo ========================================
echo Starting Both Services
echo ========================================
echo.
echo Starting API server...
start cmd /k "cd /d %CD% && call venv\Scripts\activate && cd app && python api.py"

timeout /t 5 /nobreak

echo Starting Streamlit dashboard...
start cmd /k "cd /d %CD% && call venv\Scripts\activate && streamlit run app/streamlit_app.py"

echo.
echo Both services started!
echo.
echo - API: http://localhost:8000
echo - API Docs: http://localhost:8000/docs
echo - Streamlit: http://localhost:8501
echo.
echo Close the command windows to stop the services.
echo.
pause
goto menu

:docker
echo.
echo ========================================
echo Running with Docker
echo ========================================
echo.

REM Check Docker
docker --version
if %errorlevel% neq 0 (
    echo ERROR: Docker not found. Please install Docker Desktop
    pause
    goto menu
)

echo Building and starting containers...
docker-compose up -d

echo.
echo Waiting for services to start...
timeout /t 10 /nobreak

echo.
echo Checking service status...
docker-compose ps

echo.
echo ========================================
echo Services Running!
echo ========================================
echo.
echo - API: http://localhost:8000
echo - API Docs: http://localhost:8000/docs
echo - Streamlit: http://localhost:8501
echo.
echo To stop: docker-compose down
echo To view logs: docker-compose logs -f
echo.
pause
goto menu

:tests
echo.
echo ========================================
echo Running Tests
echo ========================================
echo.

call venv\Scripts\activate
pytest tests/ -v --cov=src --cov-report=html

echo.
echo Coverage report generated in htmlcov/index.html
echo.
pause
goto menu

:docs
echo.
echo ========================================
echo Documentation
echo ========================================
echo.
echo Opening documentation files...
echo.

start README.md
start docs\PROJECT_SUMMARY.md
start docs\API.md
start docs\DEPLOYMENT.md

echo.
pause
goto menu

:exit
echo.
echo Thank you for using Customer Churn Prediction System!
echo.
exit
