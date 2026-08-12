import numpy as np
import pandas as pd
from scipy import stats
import config as cfg

# Forecast evaluation...statistical accuracy v historical average
# nested and non-nested predictive-accuracy tests
# economic value for a mean-variance investor

# Statistical accuracy

# Out-of-sample R-squared -> Positive means model beats HA
def r2_oos(actual, forecast, benchmark) -> float:
    sse_model = np.sum((actual - forecast) ** 2)
    sse_bench = np.sum((actual - benchmark) ** 2)
    return np.nan if sse_bench == 0 else 1.0 - sse_model / sse_bench


# Clark-West test
def clark_west_test(actual, forecast_restricted, forecast_unrestricted):
    e_r = actual - forecast_restricted
    e_u = actual - forecast_unrestricted
    adj = (forecast_restricted - forecast_unrestricted) ** 2
    f = e_r ** 2 - (e_u ** 2 - adj)
    se = np.std(f, ddof=1) / np.sqrt(len(f))
    if se == 0:
        return np.nan, np.nan
    stat = np.mean(f) / se
    return stat, 1 - stats.norm.cdf(stat)


# Diebold Mariano test
def diebold_mariano_test(actual, forecast_1, forecast_2, h=1,
                         alternative="two-sided"):
    d = (actual - forecast_1) ** 2 - (actual - forecast_2) ** 2
    n = len(d)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, max(1, h - 1) + 1):
        w = 1.0 - k / (max(1, h - 1) + 1)
        gamma_sum += 2 * w * np.cov(d[k:], d[:-k], ddof=1)[0, 1]
    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        return np.nan, np.nan
    stat = np.mean(d) / np.sqrt(var_d)
    if alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(stat)))
    elif alternative == "less":
        p = stats.norm.cdf(stat)
    else:
        p = 1 - stats.norm.cdf(stat)
    return stat, p


# Economic value
def cer_gain(actual, forecast, benchmark, gamma=cfg.GAMMA,
             transaction_cost=cfg.TRANSACTION_COST) -> float:
    def portfolio_return(fcst):
        sigma2 = pd.Series(actual).expanding(min_periods=2).var().shift(1).values
        first_valid = sigma2[~np.isnan(sigma2)][0] if np.any(~np.isnan(sigma2)) else 1e-4
        sigma2 = np.clip(np.where(np.isnan(sigma2), first_valid, sigma2), 1e-8, None)
        w = np.clip((1.0 / gamma) * (fcst / sigma2), 0.0, 1.5)
        r_p = w * actual
        if transaction_cost > 0:
            r_p -= transaction_cost * np.abs(np.diff(w, prepend=w[0]))
        return r_p

    r_model = portfolio_return(forecast)
    r_bench = portfolio_return(benchmark)
    cer_model = np.mean(r_model) - (gamma / 2) * np.var(r_model)
    cer_bench = np.mean(r_bench) - (gamma / 2) * np.var(r_bench)
    return (cer_model - cer_bench) * 12 * 100


# Annualized Sharpe ratio from monthly returns
def sharpe_ratio(returns) -> float:
    return 0.0 if np.std(returns) == 0 else np.mean(returns) / np.std(returns) * np.sqrt(12)
