# MLOps Pipeline - Linear Regression on California Housing Dataset

This repository contains a complete MLOps pipeline for training, testing, quantizing, and deploying a Linear Regression model on the California Housing dataset from sklearn.

## Project Overview

The pipeline includes:
- **Model Training**: Linear Regression using scikit-learn
- **Testing**: Comprehensive unit tests for data loading, model training, and performance
- **Quantization**: Manual 8-bit quantization of model parameters
- **Dockerization**: Containerized deployment
- **CI/CD**: Automated workflow with GitHub Actions

## Repository Structure

```
├── src/
│   ├── train.py          # Model training script
│   ├── quantize.py       # Model quantization script
│   ├── predict.py        # Model prediction script
│   └── utils.py          # Utility functions
├── tests/
│   └── test_train.py     # Unit tests
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
├── .github/
│   └── workflows/
│       └── ci.yml        # CI/CD workflow
└── README.md            # This file
```

## Model Performance Comparison

| Metric | Original Model | Quantized Model | Difference |
|--------|----------------|-----------------|------------|
| R² Score | 0.6469 | 0.6469 | 0.0000 |
| Mean Squared Error | 0.4627 | 0.4627 | 0.0000 |
| Model Size | 1.8 KB | 0.7 KB | -59.0% |
| Inference Speed | 0.03s | 0.03s | 0.000s |

## Usage

### Local Development

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python src/train.py
```

4. Run tests:
```bash
pytest tests/
```

5. Quantize the model:
```bash
python src/quantize.py
```

### Docker

Build and run the container:
```bash
docker build -t mlops-pipeline .
docker run mlops-pipeline
```

## CI/CD Pipeline

The GitHub Actions workflow includes three jobs:
1. **test suite**: Runs pytest to validate code quality
2. **train and quantize**: Trains model and performs quantization
3. **build and test container**: Builds Docker image and tests prediction

## Dataset

- **Source**: California Housing dataset from sklearn.datasets
- **Features**: 8 numerical features (MedInc, HouseAge, AveRooms, etc.)
- **Target**: Median house value for California districts
- **Samples**: 20,640

## Model Details

- **Algorithm**: Ridge Regression with Polynomial Features (scikit-learn)
- **Quantization**: Manual 8-bit unsigned integer quantization
- **Serialization**: joblib for model persistence
- **Performance**: R² score > 0.6 (threshold requirement)

## Requirements

- Python 3.8+
- scikit-learn
- numpy
- pandas
- joblib
- pytest
- Docker (for containerization)

## License

This project is my major exam for ml ops