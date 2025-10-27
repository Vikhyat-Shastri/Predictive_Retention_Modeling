#!/usr/bin/env python3
"""
Setup and Run Script for Customer Churn Prediction System
Cross-platform Python script for easy project management
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60 + "\n")


def print_menu():
    """Display main menu"""
    print("\nCustomer Churn Prediction System - Management Script")
    print("\n" + "=" * 60)
    print("Please select an option:")
    print("=" * 60)
    print("\n1.  Setup Environment (First Time)")
    print("2.  Train All Models")
    print("3.  Run Streamlit Dashboard")
    print("4.  Run FastAPI Server")
    print("5.  Run Both Services")
    print("6.  Run with Docker")
    print("7.  Run Tests")
    print("8.  Check Code Quality")
    print("9.  Generate Documentation")
    print("10. View Logs")
    print("11. Clean Build Files")
    print("12. Exit")
    print("\n" + "=" * 60)


def run_command(command, shell=True):
    """Run shell command"""
    try:
        subprocess.run(command, shell=shell, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def setup_environment():
    """Setup virtual environment and install dependencies"""
    print_header("Setting Up Environment")
    
    # Check Python version
    print("Checking Python version...")
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9+ required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment
    print("\nCreating virtual environment...")
    if not run_command(f"{sys.executable} -m venv venv"):
        return False
    print("✓ Virtual environment created")
    
    # Determine activation command
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip
    print("\nUpgrading pip...")
    run_command(f"{pip_cmd} install --upgrade pip")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command(f"{pip_cmd} install -r requirements.txt"):
        return False
    print("✓ Dependencies installed")
    
    print_header("Setup Complete!")
    print("\nNext steps:")
    print("1. Train models (Option 2)")
    print("2. Run applications (Options 3-6)")
    return True


def train_models():
    """Train all models"""
    print_header("Training Models")
    
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    print("Starting model training...")
    print("This may take 10-20 minutes...\n")
    
    if run_command(f"{python_cmd} train_models.py"):
        print_header("Training Complete!")
        print("\nModels saved to: models/")
        return True
    return False


def run_streamlit():
    """Run Streamlit dashboard"""
    print_header("Starting Streamlit Dashboard")
    
    if platform.system() == "Windows":
        streamlit_cmd = "venv\\Scripts\\streamlit"
    else:
        streamlit_cmd = "venv/bin/streamlit"
    
    print("Dashboard URL: http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    
    run_command(f"{streamlit_cmd} run app/streamlit_app.py")


def run_api():
    """Run FastAPI server"""
    print_header("Starting FastAPI Server")
    
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    print("API URL: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")
    
    run_command(f"{python_cmd} app/api.py")


def run_both():
    """Run both services"""
    print_header("Starting Both Services")
    
    import multiprocessing
    
    def run_api_process():
        if platform.system() == "Windows":
            python_cmd = "venv\\Scripts\\python"
        else:
            python_cmd = "venv/bin/python"
        subprocess.run(f"{python_cmd} app/api.py", shell=True)
    
    def run_streamlit_process():
        if platform.system() == "Windows":
            streamlit_cmd = "venv\\Scripts\\streamlit"
        else:
            streamlit_cmd = "venv/bin/streamlit"
        subprocess.run(f"{streamlit_cmd} run app/streamlit_app.py", shell=True)
    
    print("Starting API server...")
    api_process = multiprocessing.Process(target=run_api_process)
    api_process.start()
    
    print("Waiting for API to start...")
    time.sleep(5)
    
    print("Starting Streamlit dashboard...")
    streamlit_process = multiprocessing.Process(target=run_streamlit_process)
    streamlit_process.start()
    
    print("\n" + "=" * 60)
    print("Services Running!")
    print("=" * 60)
    print("\n- API: http://localhost:8000")
    print("- API Docs: http://localhost:8000/docs")
    print("- Streamlit: http://localhost:8501")
    print("\nPress Ctrl+C to stop all services\n")
    
    try:
        api_process.join()
        streamlit_process.join()
    except KeyboardInterrupt:
        print("\nStopping services...")
        api_process.terminate()
        streamlit_process.terminate()


def run_docker():
    """Run with Docker"""
    print_header("Running with Docker")
    
    # Check if Docker is installed
    print("Checking Docker installation...")
    if not run_command("docker --version"):
        print("ERROR: Docker not found. Please install Docker Desktop")
        return False
    
    print("\nBuilding and starting containers...")
    if not run_command("docker-compose up -d"):
        return False
    
    print("\nWaiting for services to start...")
    time.sleep(10)
    
    print("\nChecking service status...")
    run_command("docker-compose ps")
    
    print_header("Services Running!")
    print("\n- API: http://localhost:8000")
    print("- API Docs: http://localhost:8000/docs")
    print("- Streamlit: http://localhost:8501")
    print("\nTo stop: docker-compose down")
    print("To view logs: docker-compose logs -f")
    return True


def run_tests():
    """Run test suite"""
    print_header("Running Tests")
    
    if platform.system() == "Windows":
        pytest_cmd = "venv\\Scripts\\pytest"
    else:
        pytest_cmd = "venv/bin/pytest"
    
    print("Running test suite with coverage...\n")
    run_command(f"{pytest_cmd} tests/ -v --cov=src --cov-report=html --cov-report=term")
    
    print("\nCoverage report generated in: htmlcov/index.html")


def check_code_quality():
    """Run code quality checks"""
    print_header("Checking Code Quality")
    
    if platform.system() == "Windows":
        black_cmd = "venv\\Scripts\\black"
        flake8_cmd = "venv\\Scripts\\flake8"
    else:
        black_cmd = "venv/bin/black"
        flake8_cmd = "venv/bin/flake8"
    
    print("Running Black formatter check...")
    run_command(f"{black_cmd} --check src/ app/ tests/")
    
    print("\nRunning Flake8 linter...")
    run_command(f"{flake8_cmd} src/ app/ tests/ --max-line-length=120 --ignore=E203,W503")
    
    print("\n✓ Code quality checks complete")


def generate_docs():
    """Generate documentation"""
    print_header("Generating Documentation")
    
    print("Documentation files:")
    print("- README.md")
    print("- docs/PROJECT_SUMMARY.md")
    print("- docs/API.md")
    print("- docs/DEPLOYMENT.md")
    
    # Open in browser (optional)
    if input("\nOpen documentation in browser? (y/n): ").lower() == 'y':
        import webbrowser
        webbrowser.open('README.md')


def view_logs():
    """View application logs"""
    print_header("Viewing Logs")
    
    log_files = list(Path('.').glob('training_*.log')) + list(Path('.').glob('*.log'))
    
    if not log_files:
        print("No log files found")
        return
    
    print("Available log files:")
    for i, log_file in enumerate(log_files, 1):
        print(f"{i}. {log_file}")
    
    try:
        choice = int(input("\nSelect log file (number): "))
        if 1 <= choice <= len(log_files):
            print(f"\nLast 50 lines of {log_files[choice-1]}:\n")
            with open(log_files[choice-1], 'r') as f:
                lines = f.readlines()
                print(''.join(lines[-50:]))
    except (ValueError, IndexError):
        print("Invalid selection")


def clean_build():
    """Clean build files"""
    print_header("Cleaning Build Files")
    
    print("Removing build artifacts...")
    
    patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '.pytest_cache',
        'htmlcov',
        '.coverage',
        '*.log',
        'build',
        'dist',
        '*.egg-info'
    ]
    
    for pattern in patterns:
        if platform.system() == "Windows":
            run_command(f"del /s /q {pattern}", shell=True)
        else:
            run_command(f"find . -name '{pattern}' -exec rm -rf {{}} +", shell=True)
    
    print("✓ Build files cleaned")


def main():
    """Main program loop"""
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (1-12): ").strip()
            
            if choice == '1':
                setup_environment()
            elif choice == '2':
                train_models()
            elif choice == '3':
                run_streamlit()
            elif choice == '4':
                run_api()
            elif choice == '5':
                run_both()
            elif choice == '6':
                run_docker()
            elif choice == '7':
                run_tests()
            elif choice == '8':
                check_code_quality()
            elif choice == '9':
                generate_docs()
            elif choice == '10':
                view_logs()
            elif choice == '11':
                clean_build()
            elif choice == '12':
                print("\nThank you for using Customer Churn Prediction System!")
                break
            else:
                print("\nInvalid choice. Please enter a number between 1 and 12.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
