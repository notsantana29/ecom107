import os, sys
import numpy as np, pandas as pd
from scipy import stats

np.random.seed(20260727)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "results")
sys.stdout = open(f"{HERE}/run_log.txt", "a")
XL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Load the forecast file and the regime masks

fc = pd.read_csv(f"{HERE}/forecasts/oos_forecasts_xl.csv", index_col=0, parse_dates=True)
a, ha, idx = fc["actual"].values, fc["HA"].values, fc.index
# NBER dating from their sheet, aligned to the predictor months
r = pd.read_excel(f"{XL}/ml_equity_premium_data.xlsx", sheet_name="NBER_recession", index_col=0)["recession"]
nb = pd.Series(r.values, index=pd.to_datetime(r.index.astype(int).astype(str), format="%Y%m")).reindex(idx).fillna(0).values.astype(bool)
regimes = {"Full": np.ones(len(a), bool), "NBER_rec": nb, "NBER_exp": ~nb}
MODELS = ["PCR","Ridge","LASSO","ENet","PLS","OLS","RF","GBRT","XGB","NN3","Mean","Median"]

# Build the test family

# Per-month CW contribution g_t for the nested comparison of HA vs the model
def cw_contrib(f):
    return (a-ha)**2 - ((a-f)**2 - (ha-f)**2)

# The family of cells is model * variant (raw/CT) * regime
cells = []   # one dict per cell (model, variant, regime, mask)
G = {}       # (model, variant) -> g_t over all months
for m in MODELS:
    for variant in ["raw","CT"]:
        f = fc[m].values if variant=="raw" else np.maximum(fc[m].values, 0.0)
        G[(m,variant)] = cw_contrib(f)
for m in MODELS:
    for variant in ["raw","CT"]:
        g = G[(m,variant)]
        for rname, mask in regimes.items():
            cells.append({"model":m,"variant":variant,"regime":rname,"key":(m,variant,rname),"mask":mask})

# CW t-statistic and one-sided p-value within a cell
def cw_stat_p(g, mask):
    x = g[mask]; n = len(x)
    if n < 15: return np.nan, np.nan
    se = np.std(x, ddof=1)/np.sqrt(n)
    if se == 0: return np.nan, np.nan
    t = np.mean(x)/se
    return t, 1 - stats.norm.cdf(t)

# Fill in the test statistic, p-value and R2 for every cell
for c in cells:
    c["t"], c["p"] = cw_stat_p(G[(c["model"],c["variant"])], c["mask"])
    # R2 for the cell, needed to flag which cells count as positive
    f = fc[c["model"]].values if c["variant"]=="raw" else np.maximum(fc[c["model"]].values,0.0)
    mk = c["mask"]; c["R2"] = (1 - np.sum((a[mk]-f[mk])**2)/np.sum((a[mk]-ha[mk])**2))*100

df = pd.DataFrame(cells)
m_family = df["p"].notna().sum()
print(f"Family size (cells with a valid CW test): {m_family}")

# 8 cells positive + significant before any correction
pos_sig = df[(df.R2 > 0) & (df.p < 0.05)].copy().sort_values("p")
print(f"\nPositive & CW-significant at 5% (nominal): {len(pos_sig)} cells")


# Holm-Bonferroni

# Step-down FWER adjustment over the full family
pv = df["p"].dropna().values
order = np.argsort(pv)
holm = np.empty_like(pv); running = 0.0
for rank, i in enumerate(order):
    val = (m_family - rank) * pv[i]
    running = max(running, val)
    holm[i] = min(running, 1.0)
holm_map = dict(zip(df["p"].dropna().index, holm))


# Benjamini-Hochberg

# Step-up FDR adjustment over the same family
bh = np.empty_like(pv); prev = 1.0
for rank in range(len(order)-1, -1, -1):
    i = order[rank]
    val = pv[i] * m_family / (rank+1)
    prev = min(prev, val)
    bh[i] = min(prev, 1.0)
bh_map = dict(zip(df["p"].dropna().index, bh))

# Romano-Wolf

# Stepdown FWER adjustment via a circular block bootstrap
B, ell = 5000, 12
N = len(a)
active_idx = df[df["p"].notna()].index.tolist()
tvec = df.loc[active_idx, "t"].values.astype(float)
# Collect each cell's contribution series, mask and observed mean
cell_g = []; cell_mask = []; cell_mu = []
for i in active_idx:
    row = df.loc[i]; g = G[(row["model"],row["variant"])]; mk = row["mask"]
    cell_g.append(g); cell_mask.append(mk); cell_mu.append(np.mean(g[mk]))
cell_g = np.array(cell_g); cell_mask = np.array(cell_mask); cell_mu = np.array(cell_mu)
K = len(active_idx)
# Bootstrap the max-statistic distribution, recentring each cell by its observed mean so the null holds
boot_t = np.empty((B, K))
n_blocks = int(np.ceil(N/ell))
for b in range(B):
    starts = np.random.randint(0, N, size=n_blocks)
    resamp = np.concatenate([(np.arange(s, s+ell) % N) for s in starts])[:N]
    for k in range(K):
        mk = cell_mask[k][resamp]
        x = cell_g[k][resamp][mk] - cell_mu[k]
        n = len(x)
        if n < 15: boot_t[b,k] = -np.inf; continue
        se = np.std(x, ddof=1)/np.sqrt(n)
        boot_t[b,k] = np.mean(x)/se if se>0 else -np.inf
# Stepdown pass -> work from the largest observed t downwards -> enforces monotone adjusted p-values
rw_p = np.ones(K)
ord_desc = np.argsort(-tvec)
remaining = list(ord_desc)
while remaining:
    maxdist = boot_t[:, remaining].max(axis=1)
    k = remaining[0]  # largest observed t among the remaining cells
    p_adj = np.mean(maxdist >= tvec[k])
    rw_p[k] = max(p_adj, rw_p[ord_desc[list(ord_desc).index(k)-1]] if k!=ord_desc[0] else 0)
    remaining.remove(k)
rw_map = dict(zip(active_idx, rw_p))

# Report

# Adjusted p-values for the 8 positive-and-significant cells under all three corrections
print("\n=== Multiplicity-adjusted significance of the 8 positive & significant cells ===")
print(f"{'cell':28s} {'R2':>6s} {'raw p':>7s} {'Holm':>7s} {'BH-FDR':>8s} {'RW':>7s}")
for i in pos_sig.index:
    row = df.loc[i]; nm = f"{row['model']}+{row['variant']}/{row['regime']}"
    print(f"{nm:28s} {row['R2']:+6.2f} {row['p']:7.3f} {holm_map[i]:7.3f} {bh_map[i]:8.3f} {rw_map[i]:7.3f}")
print(f"\nBonferroni 5% threshold (m={m_family}): {0.05/m_family:.5f}")
print(f"Smallest raw p among the 8: {pos_sig['p'].min():.3f} (Mean+CT full)")
print(f"Survivors after Holm (FWER 5%): {int((pos_sig.index.map(holm_map).values < 0.05).sum())}")
print(f"Survivors after BH (FDR 5%):   {int((pos_sig.index.map(bh_map).values < 0.05).sum())}")
print(f"Survivors after Romano-Wolf:   {int((pos_sig.index.map(rw_map).values < 0.05).sum())}")

# Save the multiplicity-adjusted table of the 8 positive & significant cells
pd.DataFrame([
    {"cell": f"{df.loc[i,'model']}+{df.loc[i,'variant']}/{df.loc[i,'regime']}",
     "R2": round(df.loc[i,'R2'],2), "raw_p": round(df.loc[i,'p'],3),
     "Holm": round(holm_map[i],3), "BH_FDR": round(bh_map[i],3), "Romano_Wolf": round(rw_map[i],3)}
    for i in pos_sig.index
]).to_csv(f"{HERE}/multiplicity.csv", index=False)
# Additional Check -> repeat the Bonferroni threshold for a smaller family of full-sample cells
m24 = 0.05/24
print(f"\nRobustness: full-sample-only family m=24 -> Bonferroni threshold {m24:.5f}; min p 0.008 {'survives' if 0.008<m24 else 'fails'}")
