import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import statsmodels.api as sm
import regime
from evaluation import r2_oos, clark_west_test, cer_gain

# Runs extensions on reproduced forecasts

XL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results")
# route the run's diagnostics to a log file rather than the console
sys.stdout = open(f"{HERE}/run_log.txt", "w")
INDIV = ["OLS","PLS","PCR","LASSO","ENet","GBRT","RF","NN1","NN2","NN3","NN4","NN5",
         "Ridge","SVR","KNR","XGBoost"]

# Build the forecast file
# Line up the realized premium and the expanding-mean benchmark with their OOS months
pred = pd.read_excel(f"{XL}/ml_equity_premium_results.xlsx", sheet_name="ml_pred_equity_premium")
prem = pd.read_excel(f"{XL}/ml_equity_premium_data.xlsx", sheet_name="result_predictor").set_index("month")["log_equity_premium"]
full = prem.values
midx = {m: i for i, m in enumerate(prem.index)}
oos_m = pred["month"].values

actual = np.array([full[midx[m] + 1] for m in oos_m])              # premium at t+1
ha = np.array([full[: midx[m] + 1].mean() for m in oos_m])          # expanding mean through t
dates = pd.to_datetime([str(int(m)) for m in oos_m], format="%Y%m")

fc = pd.DataFrame({m: pred[m].values for m in INDIV}, index=dates)
fc = fc.rename(columns={"XGBoost": "XGB"})
fc["Mean"] = pred["Combined"].values                                # their mean-style combination
fc["Median"] = np.median(fc[[m if m != "XGBoost" else "XGB" for m in INDIV]].values, axis=1)
fc["HA"] = ha
fc["actual"] = actual
fc.to_csv(f"{HERE}/forecasts/oos_forecasts_xl.csv")


# Model comparison

# R2_OOS and Clark-West p-value for each model against the historical average
MODELS = ["OLS","Ridge","LASSO","ENet","RF","XGB","GBRT","NN3","Mean","Median"]
print("=== MODEL COMPARISON on THEIR forecasts (R2_OOS %, [CW p]) — should match published ===")
for m in MODELS:
    f = fc[m].values
    _, p = clark_west_test(actual, ha, f)
    print(f"  {m:<7} {r2_oos(actual, f, ha)*100:+.2f}  [{p:.3f}]")

# Regime split

print("\n=== REGIME on their Ridge/RF/XGB/Mean ===")
macro = regime.load_macro()
# Use Xu & Liu's own NBER dating (their NBER_recession sheet, aligned to predictor months)
# so the recession/expansion split reproduces their Table 2 exactly (Ridge -2.93 rec / -0.60 exp)
_xlrec = pd.read_excel(f"{XL}/ml_equity_premium_data.xlsx", sheet_name="NBER_recession", index_col=0)["recession"]
nber = pd.Series(_xlrec.values, index=pd.to_datetime(_xlrec.index.astype(int).astype(str), format="%Y%m"))
nlab = regime.nber_labels(dates, nber)

# R2_OOS (in %) for one model within a regime mask
def R2(m, mask): return regime.r2_oos(actual, fc[m].values, ha, mask)
print("NBER (rec/exp) — regime cut is NBER only, as in Xu & Liu Table 2:")
for m in ["PCR", "Ridge", "RF", "XGB", "Mean"]:
    print(f"  {m:<5} {R2(m,(nlab=='Recession').values):+.2f} / {R2(m,(nlab=='Expansion').values):+.2f}")

# Campbell-Thompson truncation
print("\n=== CT TRUNCATION on their models ===")
full_mask = np.ones(len(actual), bool)
for m in ["Ridge", "RF", "XGB", "Mean"]:
    raw = fc[m].values; ct = np.maximum(raw, 0)
    _, praw = clark_west_test(actual, ha, raw)
    _, pct = clark_west_test(actual, ha, ct)
    print(f"  {m:<5} raw {r2_oos(actual,raw,ha)*100:+.3f} [{praw:.3f}]   +CT {r2_oos(actual,ct,ha)*100:+.3f} [{pct:.3f}]")
# Count how many of the regime cells are positive for Ridge with the truncation applied
ridge_ct = np.maximum(fc["Ridge"].values, 0)
cells = {"Full":full_mask,"rec":(nlab=='Recession').values,"exp":(nlab=='Expansion').values}
npos = sum(regime.r2_oos(actual, ridge_ct, ha, mk) > 0 for mk in cells.values())
print(f"  Ridge+CT positive in {npos}/3 regime cells")


# Certainty-equivalent gain for a mean-variance investor using the Ridge forecasts
print("\n=== ECONOMIC VALUE ===")
print(f"  Ridge CER gain: {cer_gain(actual, fc['Ridge'].values, ha):+.2f}% ann")


# Overfitting
# Compare their in-sample R2 with the OOS R2 to size the overfitting gap
print("\n=== OVERFITTING ANATOMY (their in-sample vs their OOS) ===")
insamp = pd.read_excel(f"{XL}/ml_equity_premium_in_sample_results.xlsx", sheet_name="in_sample_performance").set_index("Unnamed: 0")["in_r_square(%)"]

# Calibration slope
def calib(m):
    return sm.OLS(actual, sm.add_constant(fc[m].values)).fit().params[1]
for m, key in [("XGB","XGBoost"), ("Ridge","Ridge")]:
    is_r2 = insamp[key]; oos = r2_oos(actual, fc[m].values, ha)*100
    print(f"  {m:<5} in-sample {is_r2:+.2f}  OOS {oos:+.2f}  gap {is_r2-oos:.2f}pp  calib {calib(m):+.3f}")

# Sample-split robustness
# Re-evaluate the reproduced forecasts over out-of-sample windows starting in 1957, 1965 and 1980
a_split, ha_split, idx_split = fc["actual"].values, fc["HA"].values, fc.index
def split_r2(f, mask): return (1 - np.sum((a_split[mask]-f[mask])**2)/np.sum((a_split[mask]-ha_split[mask])**2))*100
split_rows = []
for start in ["1957-01","1965-01","1980-01"]:
    mask = np.asarray(idx_split >= start)
    row = {"OOS_start": start[:4], "n": int(mask.sum())}
    row.update({mm: round(split_r2(fc[mm].values, mask), 2) for mm in ["PCR","Ridge","RF","XGB","Mean"]})
    split_rows.append(row)
pd.DataFrame(split_rows).to_csv(f"{HERE}/sample_splits.csv", index=False)

# Summary statistics
PRED24 = ["DP","DY","EP","SVAR","BM","NTIS","TBL","LTR","TMS","DFY","DFR","INFL",
          "MA_1_9","MA_1_12","MA_2_9","MA_2_12","MA_3_9","MA_3_12",
          "MOM_1","MOM_2","MOM_3","MOM_6","MOM_9","MOM_12"]
raw_summary = pd.read_excel(f"{XL}/ml_equity_premium_data.xlsx", sheet_name="result_predictor")
raw_summary["date"] = pd.to_datetime(raw_summary["month"].astype(int).astype(str), format="%Y%m")
raw_summary = raw_summary.set_index("date").sort_index()
def ar1(s): s = s.dropna(); return np.corrcoef(s.values[:-1], s.values[1:])[0,1]
summary_rows = []
for col in ["log_equity_premium"] + PRED24:
    s = raw_summary[col]
    name = "equity premium" if col == "log_equity_premium" else col
    summary_rows.append({"Variable": name, "Mean": round(s.mean(),4), "Std": round(s.std(),4),
                         "Min": round(s.min(),3), "Max": round(s.max(),3), "AR1": round(ar1(s),3)})
pd.DataFrame(summary_rows).to_csv(f"{HERE}/summary_stats.csv", index=False)

print("Saved forecasts/oos_forecasts_xl.csv, sample_splits.csv, summary_stats.csv")
