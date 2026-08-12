import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import regime
from evaluation import clark_west_test

# Every treatment (NBER regime cut, Campbell-Thompson truncation) applied to every model on Xu & Liu's forecasts 
# Clark-West p-value per cell and a crash-robustness check
# Reads results/forecasts/oos_forecasts_xl.csv -> writes three summary CSVs to results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results")
sys.stdout = open(f"{HERE}/run_log.txt", "a")
XL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
fc = pd.read_csv(f"{HERE}/forecasts/oos_forecasts_xl.csv", index_col=0, parse_dates=True)
a, ha, idx = fc["actual"].values, fc["HA"].values, fc.index

macro = regime.load_macro()
_r = pd.read_excel(f"{XL}/ml_equity_premium_data.xlsx", sheet_name="NBER_recession", index_col=0)["recession"]
nber = pd.Series(_r.values, index=pd.to_datetime(_r.index.astype(int).astype(str), format="%Y%m"))
nlab = regime.nber_labels(idx, nber)

MODELS = ["PCR","Ridge","LASSO","ENet","PLS","OLS","RF","GBRT","XGB","NN3","Mean","Median"]


# Cell-level helpers

# R2_OOS
def r2(f, m):
    if m.sum() == 0: return np.nan
    return (1 - np.sum((a[m]-f[m])**2)/np.sum((a[m]-ha[m])**2))*100

# Clark-West p-value 
def cwp(f, m):
    if m.sum() < 10: return np.nan
    return clark_west_test(a[m], ha[m], f[m])[1]

# R2 and CW p after dropping the 5 months
def robust(f, m):
    if m.sum() < 15: return np.nan, np.nan
    e_r, e_u = a-ha, a-f
    fterm = e_r**2 - (e_u**2 - (ha-f)**2)
    sub = np.where(m)[0]
    drop = sub[np.argsort(fterm[sub])[::-1][:5]]
    m2 = m.copy(); m2[drop] = False
    return r2(f, m2), cwp(f, m2)

# Regime cells

# Regime cut is NBER only (recession/expansion), as in Xu & Liu
# Tercile cuts (recession-probability, volatility) were dropped: their breakpoints involve look-ahead and the per-cell CW test is non-standard
CELLS = {
    "Full":      np.ones(len(a), bool),
    "NBER_rec":  (nlab=="Recession").values,
    "NBER_exp":  (nlab=="Expansion").values,
}

# Raw forecasts

# R2 and CW p-value for every model in every cell
rows = []
for mdl in MODELS:
    f = fc[mdl].values
    row = {"Model": mdl}
    for cell, mask in CELLS.items():
        row[f"{cell}_R2"] = round(r2(f, mask), 2)
        row[f"{cell}_CWp"] = round(cwp(f, mask), 3)
    rows.append(row)
raw = pd.DataFrame(rows)
raw.to_csv(f"{HERE}/all_models_regime_cw.csv", index=False)

# CT-truncated forecasts

# Same treatment with every forecast floored at zero
rows_ct = []
for mdl in MODELS:
    f = np.maximum(fc[mdl].values, 0)
    row = {"Model": mdl}
    for cell, mask in CELLS.items():
        row[f"{cell}_R2"] = round(r2(f, mask), 2)
        row[f"{cell}_CWp"] = round(cwp(f, mask), 3)
    rows_ct.append(row)
ct = pd.DataFrame(rows_ct)
ct.to_csv(f"{HERE}/all_models_regime_cw_CT.csv", index=False)

# Crash-robustness

survivors = []
for label, frame, tf in [("raw", raw, fc), ("CT", ct, None)]:
    for mdl in MODELS:
        f = fc[mdl].values if tf is not None else np.maximum(fc[mdl].values, 0)
        for cell, mask in CELLS.items():
            v = r2(f, mask); p = cwp(f, mask)
            if v > 0 and p < 0.05:
                rr, rp = robust(f, mask)
                survivors.append({"forecast": label, "Model": mdl, "cell": cell,
                                  "R2": round(v,2), "CWp": round(p,3),
                                  "R2_drop5": round(rr,2), "CWp_drop5": round(rp,3),
                                  "survives": bool(rr > 0 and rp < 0.05)})
surv = pd.DataFrame(survivors)
surv.to_csv(f"{HERE}/all_models_positive_cells_robustness.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 40)
print("=== RAW forecasts: R2 [CW p] per cell (all models) ===")
show = ["Full","NBER_rec","NBER_exp"]
for _, r in raw.iterrows():
    print(f"{r['Model']:<7} " + "  ".join(f"{c}:{r[c+'_R2']:+.2f}[{r[c+'_CWp']:.2f}]" for c in show))
print("\n=== CT truncation: R2 [CW p] per cell (all models) ===")
for _, r in ct.iterrows():
    print(f"{r['Model']:<7} " + "  ".join(f"{c}:{r[c+'_R2']:+.2f}[{r[c+'_CWp']:.2f}]" for c in show))
print("\n=== Every positive & 5%-significant cell — does it survive dropping top-5 crash months? ===")
print(surv.to_string(index=False) if len(surv) else "none")
print(f"\nOf {len(surv)} positive+significant cells, {int(surv['survives'].sum()) if len(surv) else 0} survive the crash-robustness check.")
print("Saved: all_models_regime_cw.csv, all_models_regime_cw_CT.csv, all_models_positive_cells_robustness.csv")
