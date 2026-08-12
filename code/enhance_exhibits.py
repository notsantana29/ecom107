import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluation import clark_west_test
from Perform_PT_test import PT_test

# The three headline exhibits:
# 1. Model-comparison table with success-ratio and Pesaran-Timmermann columns, plus an HA row
# 2. Crash-robustness figure 
# 3. NBER cumulative squared-error figure
# Outputs go to results/ and results/figures/.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results")
sys.stdout = open(f"{HERE}/run_log.txt", "a")
XL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FIG = f"{HERE}/figures"
GREEN, RED, NAVY, GREY = "#2e7d5b", "#b5473f", "#1f3b5c", "#888888"

# load forecasts with actual and HA columns
fc = pd.read_csv(f"{HERE}/forecasts/oos_forecasts_xl.csv", index_col=0, parse_dates=True)
a, ha, idx = fc["actual"].values, fc["HA"].values, fc.index
INDIV = ["OLS","PLS","PCR","LASSO","ENet","GBRT","RF","NN1","NN2","NN3","NN4","NN5","Ridge","SVR","KNR","XGB"]
# full-sample OOS R-squared vs the HA benchmark
def r2(f): return (1 - np.sum((a-f)**2)/np.sum((a-ha)**2))*100

# Enhanced model-comparison table

# one row per model: R2, Clark-West p, success ratio, Pesaran-Timmermann p
rows = []
for m in INDIV + ["Mean", "Median"]:
    f = fc[m].values
    _, cwp = clark_west_test(a, ha, f)
    sr, pt, ptp = PT_test(a.reshape(-1,1), f.reshape(-1,1))
    rows.append([m, round(r2(f),2), round(cwp,3), round(sr*100,2), round(ptp,3)])
# and an HA benchmark row (no CW p against itself)
sr_ha, pt_ha, ptp_ha = PT_test(a.reshape(-1,1), ha.reshape(-1,1))
rows.append(["HA (benchmark)", 0.00, np.nan, round(sr_ha*100,2), round(ptp_ha,3)])
tab = pd.DataFrame(rows, columns=["Model","R2_OOS","CW_p","SuccessRatio_pct","PT_p"])
tab.to_csv(f"{HERE}/model_comparison_enhanced.csv", index=False)
print("=== Enhanced Table 4.1 (verify vs paper: OLS SR 56.06, PCR 59.19, Ridge 58.54, HA 59.97) ===")
print(tab.to_string(index=False))

# Crash-robustness figure

rob = pd.read_csv(f"{HERE}/all_models_positive_cells_robustness.csv")
sig = rob[(rob.CWp < 0.05)].copy()   # the 8 positive+significant cells
# readable label for each cell, e.g. "Ridge+CT (rec)" or "PCR (raw)"
def lbl(r):
    base = "PCR (raw)" if (r.forecast=="raw") else f"{r.Model}+CT"
    cell = {"Full":"", "NBER_exp":" (exp)", "NBER_rec":" (rec)"}.get(r.cell, f" ({r.cell})")
    return base + cell
sig["label"] = sig.apply(lbl, axis=1)
sig = sig.sort_values("R2", ascending=True)
# paired horizontal bars: full-sample R2 vs R2 after dropping the 5 crash months
y = np.arange(len(sig)); w = 0.38
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.barh(y - w/2, sig.R2, w, color=GREEN, label="full sample")
ax.barh(y + w/2, sig.R2_drop5, w, color=RED, label="drop 5 crash months")
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(sig.label, fontsize=8)
# bold the PCR raw label — that one is Xu & Liu's own headline result
for i, r in enumerate(sig.itertuples()):
    if r.forecast=="raw" and r.Model=="PCR":
        ax.get_yticklabels()[i].set_color(NAVY); ax.get_yticklabels()[i].set_fontweight("bold")
ax.set_xlim(-2.0, 0.95)
ax.set_xlabel("$R^2_{OOS}$ (%)")
ax.set_title("Every Positive, Significant Cell Is Crash-Driven De-Risking\n8 cells positive & CW-significant — 0 survive dropping 5 crash months")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_crash_robustness.png", dpi=200); plt.close(fig)

# CSSED figure with NBER recession shading

# cumulative squared-error difference vs HA (rising = model winning)
def cssed(f): return np.cumsum((a-ha)**2 - (a-f)**2) * 1e4   # scaled for readability
series = {"PCR (raw)": (fc["PCR"].values, GREEN, "-"),
          "Ridge (raw)": (fc["Ridge"].values, NAVY, "--"),
          "Ridge+CT": (np.maximum(fc["Ridge"].values,0), RED, "-"),
          "Mean+CT": (np.maximum(fc["Mean"].values,0), "#d08b1e", "-")}
fig, ax = plt.subplots(figsize=(9, 5))
for name,(f,c,ls) in series.items():
    ax.plot(idx, cssed(f), color=c, ls=ls, lw=1.4, label=name)
ax.axhline(0, color="black", lw=0.8)  # HA benchmark
_r = pd.read_excel(f"{XL}/ml_equity_premium_data.xlsx", sheet_name="NBER_recession", index_col=0)["recession"]
nb = pd.Series(_r.values, index=pd.to_datetime(_r.index.astype(int).astype(str), format="%Y%m")).reindex(idx).fillna(0).values
inrec = False
for i in range(len(idx)):
    if nb[i]==1 and not inrec: start=idx[i]; inrec=True
    elif nb[i]==0 and inrec: ax.axvspan(start, idx[i], color=GREY, alpha=0.18); inrec=False
if inrec: ax.axvspan(start, idx[-1], color=GREY, alpha=0.18)
ax.set_ylabel("Cumulative SSE(HA) − SSE(Model)  ($\\times10^{-4}$)")
ax.set_xlabel("(rising = model beating HA; shaded = NBER recessions)")
ax.set_title("Cumulative Squared-Error Difference vs the Historical Average\nThe truncated models' entire edge accrues in discrete crash-month jumps")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_cssed.png", dpi=200); plt.close(fig)

print("\nSaved: fig3_crash_robustness.png (8 cells), fig4_cssed.png")
