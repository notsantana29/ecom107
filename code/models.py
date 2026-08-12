from itertools import product
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import config as cfg

# The forecasting models, from the historical-average benchmark up to the neural net.

# Benchmark

# Historical-average benchmark at each OOS month -> mean of all past returns
def expanding_ha_forecasts(y: np.ndarray, oos_start_idx: int) -> np.ndarray:
    n_oos = len(y) - oos_start_idx
    return np.array([np.mean(y[:oos_start_idx + i]) for i in range(n_oos)])


# Linear models

# OLS on all predictors
def ols_forecast(X_train, y_train, X_test):
    return LinearRegression().fit(X_train, y_train).predict(X_test)


# Pick the penalty by time-series cross-validation (no shuffling)
def _ts_cv_alpha(model_class, X_train, y_train, alpha_grid=None,
                 n_splits=cfg.N_CV_SPLITS, **kwargs):
    alpha_grid = cfg.ALPHA_GRID if alpha_grid is None else alpha_grid
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_mse = alpha_grid[0], np.inf
    for alpha in alpha_grid:
        folds = []
        for tr, va in tscv.split(X_train):
            m = model_class(alpha=alpha, **kwargs).fit(X_train[tr], y_train[tr])
            folds.append(np.mean((y_train[va] - m.predict(X_train[va])) ** 2))
        if np.mean(folds) < best_mse:
            best_mse, best_alpha = np.mean(folds), alpha
    return best_alpha


def ridge_forecast(X_train, y_train, X_test):
    alpha = _ts_cv_alpha(Ridge, X_train, y_train)
    return Ridge(alpha=alpha).fit(X_train, y_train).predict(X_test)


def lasso_forecast(X_train, y_train, X_test):
    alpha = _ts_cv_alpha(Lasso, X_train, y_train, max_iter=10000)
    return Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train).predict(X_test)


def elastic_net_forecast(X_train, y_train, X_test):
    tscv = TimeSeriesSplit(n_splits=cfg.N_CV_SPLITS)
    best = (cfg.ALPHA_GRID[0], 0.5)
    best_mse = np.inf
    for l1 in cfg.L1_RATIO_GRID:
        for alpha in cfg.ALPHA_GRID:
            folds = []
            for tr, va in tscv.split(X_train):
                m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000)
                m.fit(X_train[tr], y_train[tr])
                folds.append(np.mean((y_train[va] - m.predict(X_train[va])) ** 2))
            if np.mean(folds) < best_mse:
                best_mse, best = np.mean(folds), (alpha, l1)
    return ElasticNet(alpha=best[0], l1_ratio=best[1], max_iter=10000) \ .fit(X_train, y_train).predict(X_test)


# Tree-based models

# Grid search with time-series cross-validation -> returns best params
def _ts_grid_search(model_class, param_grid, X_train, y_train,
                    n_splits=cfg.N_CV_SPLITS, **fixed):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    keys, values = list(param_grid), list(param_grid.values())
    best_params, best_mse = {}, np.inf
    for combo in product(*values):
        params = dict(zip(keys, combo)); params.update(fixed)
        folds = []
        for tr, va in tscv.split(X_train):
            m = model_class(**params).fit(X_train[tr], y_train[tr])
            folds.append(np.mean((y_train[va] - m.predict(X_train[va])) ** 2))
        if np.mean(folds) < best_mse:
            best_mse, best_params = np.mean(folds), params
    return best_params


# Single CART tree — bridges the linear models and the tree ensembles
def decision_tree_forecast(X_train, y_train, X_test, cached_params=None):
    params = cached_params or _ts_grid_search(
        DecisionTreeRegressor, cfg.DT_PARAMS, X_train, y_train,
        random_state=cfg.RANDOM_SEED)
    model = DecisionTreeRegressor(**params).fit(X_train, y_train)
    return model.predict(X_test), model, params


def random_forest_forecast(X_train, y_train, X_test, cached_params=None):
    params = cached_params or _ts_grid_search(
        RandomForestRegressor, cfg.RF_PARAMS, X_train, y_train,
        random_state=cfg.RANDOM_SEED)
    model = RandomForestRegressor(**params).fit(X_train, y_train)
    return model.predict(X_test), model, params


# XGBoost with shallow trees
def xgboost_forecast(X_train, y_train, X_test, cached_params=None):
    params = cached_params or _ts_grid_search(
        xgb.XGBRegressor, cfg.XGB_PARAMS, X_train, y_train,
        random_state=cfg.RANDOM_SEED, objective="reg:squarederror", verbosity=0)
    model = xgb.XGBRegressor(**params).fit(X_train, y_train)
    return model.predict(X_test), model, params


# Neural network

# Configurable feed-forward network: stacked Linear -> ReLU -> Dropout blocks and a linear output
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_layers=(64, 32), dropout=0.5):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# Train a feed-forward net with early stopping
def _train_nn(X_train, y_train, hidden_layers=(64, 32), lr=1e-3):
    torch.manual_seed(cfg.RANDOM_SEED)
    p = cfg.NN_PARAMS
    val_size = max(1, int(0.2 * len(X_train)))
    X_tr = torch.FloatTensor(np.array(X_train[:-val_size]))
    y_tr = torch.FloatTensor(np.array(y_train[:-val_size]))
    X_val = torch.FloatTensor(np.array(X_train[-val_size:]))
    y_val = torch.FloatTensor(np.array(y_train[-val_size:]))
    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=p["batch_size"], shuffle=False)

    model = FeedForwardNet(X_train.shape[1], hidden_layers, p["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=p["weight_decay"])
    loss_fn = nn.MSELoss()

    best_loss, best_state, wait = np.inf, None, 0
    for _ in range(p["epochs"]):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(Xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, model.state_dict().copy(), 0
        else:
            wait += 1
            if wait >= p["early_stopping_patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# Feed-forward neural network -> selects the architecture/learning rate by validation loss when not using cached hyperparameters
def nn_forecast(X_train, y_train, X_test, cached_params=None):
    if cached_params is not None:
        best_hidden, best_lr = cached_params["hidden_layers"], cached_params["learning_rate"]
    else:
        best_hidden = cfg.NN_PARAMS["hidden_layers"][0]
        best_lr = cfg.NN_PARAMS["learning_rate"][0]
        best_val = np.inf
        for hidden in cfg.NN_PARAMS["hidden_layers"]:
            for lr in cfg.NN_PARAMS["learning_rate"]:
                cand = _train_nn(X_train, y_train, hidden, lr)
                val_size = max(1, int(0.2 * len(X_train)))
                X_val = torch.FloatTensor(np.array(X_train[-val_size:]))
                cand.eval()
                with torch.no_grad():
                    pred = cand(X_val).numpy()
                mse = np.mean((y_train[-val_size:] - pred) ** 2)
                if mse < best_val:
                    best_val, best_hidden, best_lr = mse, hidden, lr

    params = {"hidden_layers": best_hidden, "learning_rate": best_lr}
    model = _train_nn(X_train, y_train, best_hidden, best_lr)
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(np.array(X_test))).numpy()
    return pred, model, params


# Forecast combinations

# Equal-weighted mean of individual forecasts
def mean_combination(forecasts: dict) -> np.ndarray:
    return np.mean(np.column_stack(list(forecasts.values())), axis=1)


# Median of individual forecasts (robust to outlier models)
def median_combination(forecasts: dict) -> np.ndarray:
    return np.median(np.column_stack(list(forecasts.values())), axis=1)
