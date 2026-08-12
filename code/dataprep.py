import numpy as np
import pandas as pd
import config as cfg

# Data prep: build the target and the 14 predictors from the raw Welch-Goyal file
# This aligns X_t with r_{t+1}. Standardize within each training window.

# Convert the 'yyyymm' column to dates
def parse_yyyymm(yyyymm_col: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(yyyymm_col.astype(int).astype(str), format="%Y%m")

# Equity premium as continuously compounded excess return, log(1 + R_market) - log(1 + R_f)
def construct_equity_premium(df: pd.DataFrame) -> pd.Series:
    return np.log(1 + df["CRSP_SPvw"]) - np.log(1 + df["Rfree"])

# Construct the 14 standard predictors from the raw columns
def construct_predictors(df: pd.DataFrame) -> pd.DataFrame: 
    p = pd.DataFrame(index=df.index)
    p["dp"] = np.log(df["D12"]) - np.log(df["Index"])            # dividend-price
    p["dy"] = np.log(df["D12"]) - np.log(df["Index"].shift(1))   # dividend yield
    p["ep"] = np.log(df["E12"]) - np.log(df["Index"])            # earnings-price
    p["de"] = np.log(df["D12"]) - np.log(df["E12"])              # payout ratio
    p["svar"] = df["svar"]                                       # stock variance
    p["bm"] = df["b/m"]                                          # book-to-market
    p["ntis"] = df["ntis"]                                       # net equity expansion
    p["tbl"] = df["tbl"]                                         # T-bill rate
    p["lty"] = df["lty"]                                         # long-term yield
    p["ltr"] = df["ltr"]                                         # long-term return
    p["tms"] = df["lty"] - df["tbl"]                             # term spread
    p["dfy"] = df["BAA"] - df["AAA"]                             # default yield spread
    p["dfr"] = df["corpr"] - df["ltr"]                           # default return spread
    p["infl"] = df["infl"].shift(1)                              # inflation (lagged 1m)
    return p

# Return predictors and the t+1 target in one date-indexed frame
def clean_welch_goyal(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["date"] = parse_yyyymm(df["yyyymm"])
    df = df.set_index("date").sort_index()
    df["equity_premium"] = construct_equity_premium(df)

    predictors = construct_predictors(df)
    predictors[cfg.TARGET] = df["equity_premium"]

    predictors = predictors.dropna()
    predictors = predictors.loc[cfg.SAMPLE_START:cfg.SAMPLE_END]
    print(f"[dataprep] sample: {predictors.index[0]:%Y-%m} to "
          f"{predictors.index[-1]:%Y-%m}, {len(predictors)} obs")
    return predictors

# Align X_t with r_{t+1} so predictors forecast the next-month return...avoid look ahead!
def create_forecast_dataset(df: pd.DataFrame) -> pd.DataFrame:
    X = df[cfg.PREDICTORS]
    y = df[cfg.TARGET].shift(-1)
    aligned = pd.concat([X, y.rename("target")], axis=1).dropna()
    print(f"[dataprep] forecast dataset: {len(aligned)} obs, "
          f"{len(cfg.PREDICTORS)} predictors")
    return aligned

# Load dataset
def load_forecast_dataset() -> pd.DataFrame:
    processed = cfg.PROCESSED_DATA_DIR / "forecast_dataset.csv"
    if processed.exists():
        df = pd.read_csv(processed, index_col=0, parse_dates=True)
        return df
    raw = pd.read_excel(cfg.RAW_DATA_DIR / "PredictorData.xlsx", sheet_name=0)
    return create_forecast_dataset(clean_welch_goyal(raw))

# Standardize features using training-sample statistics only...no test-period information leaks into the scaling
def standardize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = np.clip(X_train.std(axis=0), 1e-8, None)
    return (X_train - mean) / std, (X_test - mean) / std
