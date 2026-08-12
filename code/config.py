from pathlib import Path
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "data"   # code/ -> repo root -> data/
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR

# Data sources
# Monthly predictor dataset
WELCH_GOYAL_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG/export?format=xlsx"
)
# NBER recession indicator (monthly)
FRED_RECESSION_SERIES = "USREC"

# Sample definition
SAMPLE_START = "1926-12"
SAMPLE_END = "2023-12"

# Out-of-sample split: first OOS forecast 1957-01
INITIAL_TRAIN_START = "1926-12"
INITIAL_TRAIN_END = "1956-12"
OOS_START = "1957-01"

# Alternative OOS start dates for robustness
OOS_ROBUSTNESS = {"1965": "1965-01", "1980": "1980-01"}

WINDOW_TYPE = "expanding"

# Re-tune ML hyperparameters every N out-of-sample months (annual)
RETUNE_FREQUENCY = 12

# Predictors (14 variables)
PREDICTORS = [
    "dp", "dy", "ep", "de", "svar", "bm", "ntis","tbl", "lty", "ltr", "tms", "dfy", "dfr", "infl"]

# Log excess market return
TARGET = "equity_premium" 

# Models
MODEL_NAMES = [
    "HA", "OLS", "Ridge", "LASSO", "ENet",
    "DT", "RF", "XGB", "NN", "Mean", "Median",
]

# Hyperparameter grids (expanding-window CV)
ALPHA_GRID = np.logspace(-4, 4, 50)
L1_RATIO_GRID = [0.25, 0.5, 0.75]
N_CV_SPLITS = 5
DT_PARAMS = {"max_depth": [1, 2, 3, 5], "min_samples_leaf": [5, 10, 20]}
RF_PARAMS = {
    "n_estimators": [300],
    "max_depth": [3, 5],
    "max_features": ["sqrt", 0.5],
    "min_samples_leaf": [5, 10],
}
XGB_PARAMS = {
    "n_estimators": [100, 300],
    "max_depth": [1, 3],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [1.0],
}
NN_PARAMS = {
    "hidden_layers": [(32,), (64, 32)],
    "learning_rate": [1e-3, 1e-4],
    "epochs": 100,
    "batch_size": 64,
    "early_stopping_patience": 10,
    "dropout": 0.5,
    "weight_decay": 1e-4,
}

# Evaluation
GAMMA = 5 # risk aversion for certainty-equivalent returns
TRANSACTION_COST = 0.0
SIGNIFICANCE_LEVEL = 0.05

# Regime analysis (recession probability)
MA_TMS_WINDOW = 36 # months in the term-spread moving average
NBER_DELAY = 24 # NBER publication delay assumed in the probit

# Reproducibility
RANDOM_SEED = 42
