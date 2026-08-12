import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import config as cfg

# Regime analysis for the OOS forecasts

# Data

# Load the monthly predictors and derive term spread, moving average, and stock variance
def load_macro() -> pd.DataFrame:
    m = pd.read_excel(cfg.RAW_DATA_DIR / "PredictorData.xlsx", sheet_name="Monthly")
    m["date"] = pd.to_datetime(m["yyyymm"].astype(str), format="%Y%m")
    m = m.set_index("date").sort_index()
    m["tms"] = m["lty"] - m["tbl"]
    m["ma_tms"] = m["tms"].rolling(cfg.MA_TMS_WINDOW).mean()
    return m


# Load monthly NBER recession indicator (1 -> recession)
def load_nber() -> pd.Series:
    nber = pd.read_csv(cfg.RAW_DATA_DIR / "nber_recessions.csv")
    date_col = [c for c in nber.columns if "date" in c.lower()][0]
    rec_col = [c for c in nber.columns if c != date_col][0]
    nber["date"] = pd.to_datetime(nber[date_col])
    return nber.set_index("date")[rec_col]


# Metrics

# Out-of-sample R-squared vs the historical average
def r2_oos(actual, forecast, ha, mask) -> float:
    if mask.sum() == 0:
        return np.nan
    a, f, h = actual[mask], forecast[mask], ha[mask]
    return (1 - np.sum((a - f) ** 2) / np.sum((a - h) ** 2)) * 100


# One-sided p-value within a mask (NaN if degenerate)
def clark_west_p(actual, forecast, ha, mask) -> float:
    if mask.sum() < 10:
        return np.nan
    a, f, h = actual[mask], forecast[mask], ha[mask]
    fhat = (a - h) ** 2 - ((a - f) ** 2 - (h - f) ** 2)
    sd = fhat.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return 1 - norm.cdf(fhat.mean() / (sd / np.sqrt(len(fhat))))


# Forecast truncation
def ct_truncate(forecast) -> np.ndarray:
    return np.maximum(forecast, 0)


# Recession probability (recursive probit)

# Estimate the recursive recession probability for each OOS month
def recession_probabilities(oos_dates, macro, nber) -> pd.Series:
    na = nber.reindex(macro.index, method="ffill").fillna(0)
    probs = pd.Series(np.nan, index=oos_dates, dtype=float)
    for date in oos_dates:
        if date not in macro.index:
            continue
        tms, ma_tms = macro.loc[date, "tms"], macro.loc[date, "ma_tms"]
        if pd.isna(tms) or pd.isna(ma_tms):
            continue
        available = na.loc[:date - pd.DateOffset(months=cfg.NBER_DELAY)]
        if len(available) < 60:
            continue
        X = macro.loc[available.index, ["tms", "ma_tms"]].dropna()
        y = na.shift(-1).reindex(X.index).dropna()
        common = X.index.intersection(y.index)
        if len(common) < 60 or y.loc[common].nunique() < 2:
            continue
        try:
            res = sm.Probit(y.loc[common].values.astype(int),
                            sm.add_constant(X.loc[common].values, has_constant="add")
                            ).fit(disp=0, maxiter=200)
            probs.loc[date] = float(res.predict(np.r_[1.0, tms, ma_tms].reshape(1, -1))[0])
        except Exception:
            continue
    return probs


# Label each observation Low / Medium / High
# Full-sample -> breakpoints are 1/3 and 2/3 quantiles
# Recursive -> breakpoints use only the values observed strictly before each month
def terciles(series: pd.Series, recursive: bool = False, burn_in: int = 60) -> pd.Series:
    idx = series.index
    if not recursive:
        edges = series.dropna().quantile([1 / 3, 2 / 3]).values
        lab = pd.Series("Medium", index=idx)
        lab[series <= edges[0]] = "Low"
        lab[series > edges[1]] = "High"
        lab[series.isna()] = np.nan
        return lab

    values = series.values
    lab = pd.Series("burn-in", index=idx, dtype=object)
    for i in range(len(values)):
        if np.isnan(values[i]):
            continue
        hist = values[:i][~np.isnan(values[:i])]
        if len(hist) < burn_in:
            continue
        q33, q67 = np.quantile(hist, [1 / 3, 2 / 3])
        lab.iloc[i] = "Low" if values[i] <= q33 else ("High" if values[i] > q67 else "Medium")
    return lab


# Terciles of realized stock variance over the OOS window
def volatility_terciles(oos_dates, macro) -> pd.Series:
    svar = macro["svar"].reindex(oos_dates)
    edges = svar.dropna().quantile([1 / 3, 2 / 3]).values
    lab = pd.Series("Medium", index=oos_dates)
    lab[svar <= edges[0]] = "Low"
    lab[svar > edges[1]] = "High"
    lab[svar.isna()] = np.nan
    return lab


# Recession / expansion label per OOS month
def nber_labels(oos_dates, nber) -> pd.Series:
    aligned = nber.reindex(oos_dates, method="ffill").fillna(0)
    return aligned.map({1: "Recession", 0: "Expansion"})
