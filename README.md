# Market Regime Detection System

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
