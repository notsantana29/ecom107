# Can Machine Learning Beat the Historical Average?

A replication and extension of Xu & Liu (2024), forecasting the US equity premium out of sample.

## Repository layout

- `code/` — the analysis pipeline (Python), plus the shared model and evaluation library
- `data/` — Xu & Liu's Excel files, the Welch–Goyal raw predictor data, and the reproduced forecasts
- `results/` — the generated result CSVs and figures

## Running the analysis

Everything runs on Python 3 (see `requirements.txt`).

```
pip install -r requirements.txt

python code/extend_on_xl.py        # Model comparisons
python code/extend_all_models.py   # NBER Recession/Expansion split + Campbell–Thompson truncation + crash-robustness
python code/multiple_testing.py    # Holm–Bonferroni, Benjamini–Hochberg and Romano–Wolf corrections
python code/generate_figures.py    # Model-comparison, Overfitting-scatter and Variable-importance figures 
python code/enhance_exhibits.py    # Enhanced comparison table and the crash-robustness and CSSED figures
```

## Data flow

Xu & Liu's Excel files (`data/`) -> `extend_on_xl.py` -> `results/forecasts/oos_forecasts_xl.csv` ->
the extension scripts -> result CSVs in `results/` -> the figure generators -> the exhibits

## Library versions

The model forecasts used here are Xu & Liu's own reproduced numbers (read from `data/`)
Tresults do not depend on your installed library versions
This code does not switch versions at runtime

The one version-dependent exhibit is the library-robustness check in `data/version_sensitivity.csv`, which lists each model's out-of-sample R^2 under three settings:

- `Published`             -> Xu & Liu's published values
- `Matched_py3810_xgb176` -> legacy environment: Python 3.8.10, scikit-learn 1.3.2, XGBoost 1.7.6
- `Modern_py314_xgb320`   -> current environment: Python 3.14, scikit-learn 1.8, XGBoost 3.2

Those columns were produced by installing each environment separately and re-running the model-training
code, then recording the numbers into this file. `generate_figures.py` reads it.

Legacy environment:
- Python 3.8.10
- numpy 1.24.4
- pandas 2.0.3
- scipy 1.10.1
- scikit-learn 1.3.2
- xgboost 1.7.6
- torch 2.0.1
- skorch 0.15.0
- openpyxl 3.1.5

Current environment:
- Python 3.14.3
- numpy 2.4.4
- pandas 3.0.2
- scipy 1.17.1
- scikit-learn 1.8.0
- xgboost 3.2.0
- torch 2.12.0
- skorch 1.4.0
- statsmodels 0.14.6
- matplotlib 3.10.8
- openpyxl 3.1.5

statsmodels and matplotlib appear only in the current environment; the legacy environment is used solely to reproduce Xu & Liu's model forecasts.

## Source

This study replicates and extends Xu, Xingfu, and Wei-han Liu. "Forecasting the equity premium: can machine learning beat the historical average?" (2024) https://doi.org/10.1080/14697688.2024.2409278

Original data and code: https://github.com/XingfuXu/EquityPremiumPrediction-Jupyter

