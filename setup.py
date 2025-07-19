from setuptools import setup, find_packages

setup(
    name="predictive_retention_modeling",
    version="0.1.0",
    description="A modular, scalable, recruiter-friendly, and production-ready predictive retention modeling package.",
    author="Shastri-727",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Dependencies are managed in requirements.txt
    ],
    include_package_data=True,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "predict-retention=src.inference:main"
        ]
    },
)
