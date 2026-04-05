from pathlib import Path


def ensure_directory(path: Path) -> None:
    if path.exists():
        print(f"Directory already exists: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")


def write_file(path: Path, content: str = "") -> None:
    if path.exists():
        print(f"File already exists: {path}")
    else:
        path.write_text(content, encoding="utf-8")
        print(f"Created file: {path}")


def main() -> None:
    root = Path(__file__).resolve().parent

    directories = [
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "labels",
        root / "models" / "checkpoints",
        root / "training",
        root / "inference",
        root / "dashboard",
        root / "notebooks",
        root / "tests",
        root / "models",
    ]

    init_packages = [
        root / "models" / "__init__.py",
        root / "training" / "__init__.py",
        root / "inference" / "__init__.py",
        root / "dashboard" / "__init__.py",
    ]

    gitignore_content = """venv/
__pycache__/
*.pyc
data/raw/
data/processed/
models/checkpoints/
.env
*.pt
.DS_Store
mlruns/
"""

    requirements_content = """torch
yfinance
pandas
numpy
scikit-learn
hmmlearn
plotly
streamlit
mlflow
fredapi
tqdm
matplotlib
seaborn
scipy
"""

    readme_content = """# Market Regime Detection System

## Overview
This project builds a Market Regime Detection System in Python using LSTM-based sequence modeling, BAM associative memory for regime retrieval support, and Markov smoothing to stabilize regime transitions over time.

## Setup
1. Create and activate a virtual environment.
2. Install the dependencies listed in `requirements.txt`.
3. Run the training and inference modules as needed for your workflow.

## Project Structure
- `data/raw`: Raw market data inputs.
- `data/processed`: Cleaned and feature-engineered datasets.
- `data/labels`: Regime labels and annotation files.
- `models/checkpoints`: Saved model checkpoints.
- `training`: Training code and experiment logic.
- `inference`: Inference and prediction scripts.
- `dashboard`: Streamlit dashboard components.
- `notebooks`: Exploration and prototyping notebooks.
- `tests`: Automated tests.

## Usage
Train models from the training pipeline, generate regime predictions with the inference package, and launch the dashboard to inspect detected market states and transitions.
"""

    for directory in directories:
        ensure_directory(directory)

    for package_file in init_packages:
        write_file(package_file)

    write_file(root / ".gitignore", gitignore_content)
    write_file(root / "requirements.txt", requirements_content)
    write_file(root / "README.md", readme_content)


if __name__ == "__main__":
    main()
