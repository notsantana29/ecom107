import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results")
sys.stdout = open(f"{HERE}/run_log.txt", "a")
XL = os.path.join(ROOT, "data")
FIG = f"{HERE}/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
GREEN, RED, NAVY = "#2e7d5b", "#b5473f", "#1f3b5c"

# Model comparison

# regime table is only needed for the Median combination's full-sample R2
reg = pd.read_csv(f"{HERE}/all_models_regime_cw.csv")
# published R2s from the version-sensitivity table, with the Median appended
vs = pd.read_csv(f"{XL}/version_sensitivity.csv")[["Model","Published"]]
vs = vs.rename(columns={"Published":"Full_R2"}).replace({"XGBoost":"XGB"})
med = pd.DataFrame({"Model":["Median"], "Full_R2":[float(reg.loc[reg.Model=="Median","Full_R2"].iloc[0])]})
d = pd.concat([vs, med], ignore_index=True).sort_values("Full_R2")

# horizontal bars, green for positive and red for negative, with the value printed next to each bar
fig, ax = plt.subplots(figsize=(7, 6))
colors = [GREEN if v > 0 else RED for v in d.Full_R2]
ax.barh(d.Model, d.Full_R2, color=colors)
ax.axvline(0, color=NAVY, lw=1)
ax.set_xlabel("Out-Of-Sample $R^2$ vs Historical Average (%)")
ax.set_title("No Model Beats the Historical Average\n(PCR the sole marginal exception, +0.23%)")
for y, (m, v) in enumerate(zip(d.Model, d.Full_R2)):
    ax.text(v + (0.6 if v > 0 else -0.6), y, f"{v:+.2f}", va="center",
            ha="left" if v > 0 else "right", fontsize=7)
ax.set_xlim(-46, 8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_model_comparison.png", dpi=200); plt.close(fig)

# IS vs OOS overfitting scatter
insamp = pd.read_excel(f"{XL}/ml_equity_premium_in_sample_results.xlsx",
                       sheet_name="in_sample_performance").set_index("Unnamed: 0")["in_r_square(%)"]
oos = pd.read_csv(f"{XL}/version_sensitivity.csv").set_index("Model")["Published"]
models = [m for m in insamp.index if m in oos.index and m != "HA"]
xs = insamp.loc[models].values; ys = oos.loc[models].values

fig, ax = plt.subplots(figsize=(7.5, 6))
# color each model by family so the clusters are readable
def cls(m):
    if m in ("Ridge","LASSO","ENet","PCR","PLS","OLS"): return GREEN   # linear/shrinkage
    if m in ("GBRT","RF","XGBoost"): return RED                        # trees
    return NAVY                                                        # NN/SVR/KNR
ax.scatter(xs, ys, c=[cls(m) for m in models], s=48, zorder=3)
# 45-degree line: points on it would generalize perfectly; everything sits below it
lo = min(xs.min(), ys.min()) - 5; hi = max(xs.max(), ys.max()) + 5
ax.plot([lo, hi], [lo, hi], color="grey", ls=":", lw=1)
ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
# manual label offsets (data units) to declutter the near-origin shrinkage cluster
OFF = {"PCR":(1.3,1.2),"LASSO":(1.3,-0.2),"ENet":(1.3,-1.6),"Ridge":(-6.5,1.2),"PLS":(1.3,-0.4),
       "NN2":(-6.0,0.4),"NN1":(-6.0,-0.4),"NN3":(1.3,-0.6),"NN4":(1.3,1.0),"NN5":(1.3,0.6),
       "OLS":(1.3,0.6),"RF":(1.3,0.8),"GBRT":(1.3,-1.8),"KNR":(1.3,-1.8),"SVR":(1.3,0.6),
       "XGBoost":(-9.5,-0.4)}
for m, x, y in zip(models, xs, ys):
    dx, dy = OFF.get(m, (1.3, 0.6))
    ax.annotate(m, (x, y), (x+dx, y+dy), fontsize=7)
ax.set_xlabel("In-Sample $R^2$ (%)"); ax.set_ylabel("Out-Of-Sample $R^2_{OOS}$ (%)")
lim = [lo, hi]
ax.set_title("Overfitting Generalizes Across Models\nHigh in-sample fit does not carry out of sample; XGBoost is the extreme case")
ax.set_xlim(lim); ax.set_ylim(lim)
handles = [Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=7,label=l)
           for c,l in [(GREEN,"linear/shrinkage"),(RED,"tree ensembles"),(NAVY,"NN / SVR / KNR")]]
ax.legend(handles=handles + [Line2D([0],[0],color='grey',ls=':',label='45° line')], loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_overfitting_gap.png", dpi=200); plt.close(fig)

# Variable importance for Ridge and RF
vr = pd.read_excel(f"{XL}/ml_equity_premium_results.xlsx", sheet_name="variable_impt_r2_reduction", index_col=0)
MACRO = ["DP","DY","EP","SVAR","BM","NTIS","TBL","LTR","TMS","DFY","DFR","INFL"]
# six most important predictors for a model
def top6(model):
    s = vr[model].sort_values(ascending=False).head(6)
    return s.index[::-1], s.values[::-1]
# side-by-side panels, green for macro predictors and amber for technicals
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, mdl in zip(axes, ["Ridge", "RF"]):
    names, vals = top6(mdl)
    cols = [GREEN if n in MACRO else "#d08b1e" for n in names]
    ax.barh(range(len(names)), vals, color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_title(f"{mdl}: top-6 predictors", fontsize=10)
    ax.set_xlabel("$R^2$ Reduction When Removed (%)", fontsize=8)
handles = [Line2D([0],[0],marker='s',color='w',markerfacecolor=GREEN,markersize=8,label='macro'),
           Line2D([0],[0],marker='s',color='w',markerfacecolor="#d08b1e",markersize=8,label='technical')]
fig.suptitle("Variable Importance (Xu & Liu's $R^2$-Reduction Measure)\nRidge leans on interest-rate / default macros; RF also draws on technical moving averages",
             y=1.06, fontsize=10)
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0,0.05,1,0.90]); fig.savefig(f"{FIG}/fig5_varimportance.png", dpi=200, bbox_inches="tight"); plt.close(fig)

print("Saved:", os.listdir(FIG))
