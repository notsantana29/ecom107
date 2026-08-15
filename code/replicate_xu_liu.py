import os, gc, time
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import sklearn, xgboost
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
VS_CSV = os.path.join(DATA, "version_sensitivity.csv")

# Environment Detection
skl, xgbv = sklearn.__version__, xgboost.__version__
def _mm(v): return tuple(int(x) for x in v.split(".")[:2])
if _mm(skl) <= (1, 3) and xgbv.startswith("1."):
    COL = "Matched_py3810_xgb176"
elif _mm(skl) >= (1, 8) and xgbv.startswith("3."):
    COL = "Modern_py314_xgb320"
else:
    COL = f"Env_skl{skl}_xgb{xgbv}"
print(f"Environment: scikit-learn {skl}, xgboost {xgbv}  ->  column '{COL}'")

# Data COnstruction
pdf = pd.read_excel(os.path.join(DATA, "ml_equity_premium_data.xlsx"), sheet_name="result_predictor")
p0 = pdf.drop(["month", "equity_premium"], axis=1)
X = np.concatenate([p0["log_equity_premium"][1:].values.reshape(-1, 1), p0.iloc[0:(p0.shape[0] - 1), 1:]], axis=1)
N, ncols = X.shape[0], X.shape[1]
actual = X[:, [0]]
ha = (p0["log_equity_premium"].values[0:N].cumsum() / np.arange(1, N + 1)).reshape(-1, 1)
oos = pdf.index[pdf["month"] == 195701][0]
actual_oos, ha_oos = actual[oos:], ha[oos:]
MSFE_HA = mean_squared_error(ha_oos, actual_oos)

preds = {m: [] for m in ["OLS","PLS","PCR","LASSO","ENet","GBRT","RF","Ridge","SVR","KNR","XGBoost"]}
fit = {}
mi, t0 = 1, time.time()

for t in range(oos, N):
    Xa, ya = X[:t, 1:ncols], X[:t, 0]
    s = int(len(Xa) * 0.85)
    Xtr, Xva, ytr, yva = Xa[:s], Xa[s:t], ya[:s], ya[s:t]
    Xte = X[[t], 1:ncols]

    if mi % 12 == 1:
        mi += 1; gc.collect()
        m = LinearRegression().fit(Xa, ya); fit["OLS"] = m; preds["OLS"].append(m.predict(Xte)[0])

        bm, bk = float("inf"), 1
        for k in range(1, 9):
            if k > min(Xtr.shape): continue
            e = mean_squared_error(PLSRegression(n_components=k).fit(Xtr, ytr).predict(Xva).flatten(), yva)
            if e < bm: bm, bk = e, k
        m = PLSRegression(n_components=bk).fit(Xa, ya); fit["PLS"] = m
        preds["PLS"].append(float(m.predict(Xte).flatten()[0]))

        bm, bk = float("inf"), 1
        for k in range(1, 9):
            if k > min(Xtr.shape): continue
            pc = PCA(n_components=k).fit(Xtr); rg = LinearRegression().fit(pc.transform(Xtr), ytr)
            e = mean_squared_error(rg.predict(pc.transform(Xva)), yva)
            if e < bm: bm, bk = e, k
        pc = PCA(n_components=bk).fit(Xa); rg = LinearRegression().fit(pc.transform(Xa), ya)
        fit["PCR"] = (pc, rg); preds["PCR"].append(rg.predict(pc.transform(Xte))[0])

        bm, ba = float("inf"), 0.01
        for a in 10 ** np.arange(-4, 1.001, 0.2):
            e = mean_squared_error(Lasso(alpha=a).fit(Xtr, ytr).predict(Xva), yva)
            if e < bm: bm, ba = e, a
        m = Lasso(alpha=ba).fit(Xa, ya); fit["LASSO"] = m; preds["LASSO"].append(m.predict(Xte)[0])

        bm, bp = float("inf"), {}
        for a in 10 ** np.arange(-4, 1.001, 0.2):
            for l1 in np.arange(0.2, 1, 0.3):
                e = mean_squared_error(ElasticNet(alpha=a, l1_ratio=l1).fit(Xtr, ytr).predict(Xva), yva)
                if e < bm: bm, bp = e, {"alpha": a, "l1_ratio": l1}
        m = ElasticNet(**bp).fit(Xa, ya); fit["ENet"] = m; preds["ENet"].append(m.predict(Xte)[0])

        bm, bp = float("inf"), {}
        for p in ParameterGrid({"n_estimators":[10,50,100,150,200],"max_depth":[2,3,4],"min_samples_leaf":[1,3,5]}):
            e = mean_squared_error(GradientBoostingRegressor(**p, random_state=1).fit(Xtr, ytr).predict(Xva), yva)
            if e < bm: bm, bp = e, p
        m = GradientBoostingRegressor(**bp, random_state=1).fit(Xa, ya); fit["GBRT"] = m
        preds["GBRT"].append(m.predict(Xte)[0])

        bm, bp = float("inf"), {}
        for p in ParameterGrid({"n_estimators":[10,50,100,150,200],"max_depth":[2,3,4],"min_samples_leaf":[1,3,5]}):
            e = mean_squared_error(RandomForestRegressor(**p, random_state=1, n_jobs=1).fit(Xtr, ytr).predict(Xva), yva)
            if e < bm: bm, bp = e, p
        m = RandomForestRegressor(**bp, random_state=1, n_jobs=1).fit(Xa, ya); fit["RF"] = m
        preds["RF"].append(m.predict(Xte)[0])

        bm, ba = float("inf"), 1.0
        for a in 10 ** np.arange(0, 20.001, 1):
            e = mean_squared_error(Ridge(alpha=a).fit(Xtr, ytr).predict(Xva), yva)
            if e < bm: bm, ba = e, a
        m = Ridge(alpha=ba).fit(Xa, ya); fit["Ridge"] = m; preds["Ridge"].append(m.predict(Xte)[0])

        bm, bp = float("inf"), {}
        for p in ParameterGrid({"kernel":["linear","poly","rbf","sigmoid"],"degree":[2,3,4],"C":[0.1,0.5,1]}):
            e = mean_squared_error(SVR(**p).fit(Xtr, ytr).predict(Xva), yva)
            if e < bm: bm, bp = e, p
        m = SVR(**bp).fit(Xa, ya); fit["SVR"] = m; preds["SVR"].append(m.predict(Xte)[0])

        bm, bp = float("inf"), {}
        for p in ParameterGrid({"n_neighbors":[3,4,5,6,7],"weights":["distance","uniform"],"leaf_size":[20,30,40],"p":[1,2,3]}):
            e = mean_squared_error(KNeighborsRegressor(**p).fit(Xtr, ytr).predict(Xva), yva)
            if e < bm: bm, bp = e, p
        m = KNeighborsRegressor(**bp).fit(Xa, ya); fit["KNR"] = m; preds["KNR"].append(m.predict(Xte)[0])

        bm, bp = float("inf"), {}
        for p in ParameterGrid({"max_depth":[4,5,6,7,8],"eta":[0.01,0.1],"lambda":[0,0.5,1],"alpha":[0,0.5,1]}):
            e = mean_squared_error(XGBRegressor(**p, verbosity=0, random_state=1, n_jobs=1).fit(Xtr, ytr).predict(Xva), yva)
            if e < bm: bm, bp = e, p
        m = XGBRegressor(**bp, verbosity=0, random_state=1, n_jobs=1).fit(Xa, ya); fit["XGBoost"] = m
        preds["XGBoost"].append(m.predict(Xte)[0])
    else:
        mi += 1
        preds["OLS"].append(fit["OLS"].predict(Xte)[0])
        preds["PLS"].append(float(fit["PLS"].predict(Xte).flatten()[0]))
        pc, rg = fit["PCR"]; preds["PCR"].append(rg.predict(pc.transform(Xte))[0])
        for k in ["LASSO","ENet","GBRT","RF","Ridge","SVR","KNR","XGBoost"]:
            preds[k].append(fit[k].predict(Xte)[0])

    i, n = t - oos, N - oos
    if (i + 1) % 60 == 0 or i == 0 or i == n - 1:
        print(f"  {i+1}/{n} | {(time.time()-t0)/60:.1f}min")

r2 = {k: round((1 - mean_squared_error(np.array(v).reshape(-1, 1), actual_oos) / MSFE_HA) * 100, 2)
      for k, v in preds.items()}
print("\nModel        R2_OOS(%)")
for k, v in r2.items():
    print(f"  {k:9s} {v:+7.2f}")

if os.path.exists(VS_CSV):
    vs = pd.read_csv(VS_CSV).set_index("Model")
else:
    vs = pd.DataFrame(index=list(r2.keys())); vs.index.name = "Model"
for k, v in r2.items():
    vs.loc[k, COL] = v
if {"Matched_py3810_xgb176", "Modern_py314_xgb320"} <= set(vs.columns):
    vs["Modern_minus_Matched"] = (vs["Modern_py314_xgb320"] - vs["Matched_py3810_xgb176"]).round(2)
vs.reset_index().to_csv(VS_CSV, index=False)
print(f"\nWrote column '{COL}' to {VS_CSV}")
