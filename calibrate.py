"""Fit the 1R1C house to one real building: the ZEB Living Lab (Trondheim, NO),
dataset 2 of Sartori, ..., Knudsen et al. (2023), Data in Brief 48:109149,
data doi:10.17632/xztfbtsgys.3 (CC BY 4.0). File used: ds2-minimal-PT30M-including-forecasts.csv.

    T[k+1] - T[k] = a*(Tout[k] - T[k]) + b*Q[k] + c*I[k]      (hourly, least squares, no intercept)
    a = DT/(RC), b = DT/C, c = A_sol*DT/(1000*C)

T = ventilation extract-air temperature (the dataset paper's indoor temperature),
Q = radiator heat (heat meter) + estimated internal gains (dummies/appliances), kW,
I = satellite global horizontal irradiance (in the dataset), W/m2.
The split was fixed BEFORE any fit: train on the PRBS excitation phase (10-28 Feb 2019),
validate open-loop on the economic-MPC phase (1-13 Mar 2019).

    .venv/bin/python -I calibrate.py   (-I: the data folder is untrusted; nothing from it is imported)
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "data_external" / "knudsen6" / "ds2-minimal-PT30M-including-forecasts.csv"
OUT = HERE / "results"
SPLIT = pd.Timestamp("2019-03-01", tz="UTC")
DT = 1.0
ASSUMED = {"R": 1 / 0.15, "C": 10.0, "A_SOL": 2.0}   # copy of house.ASSUMED; HOUSE=calibrated MUST NOT change it


def load():
    d = pd.read_csv(SRC, usecols=["Time", "Tout", "PhRad_actual_power", "QestInt", "Tv_ext_ahu", "Ps"],
                    parse_dates=["Time"]).set_index("Time")
    h = d.resample("1h").mean()           # hours inside the 18-21 Feb power cut stay NaN
    return pd.DataFrame({"T": h["Tv_ext_ahu"], "t_out": h["Tout"],
                         "Q": (h["PhRad_actual_power"] + h["QestInt"]) / 1000, "solar": h["Ps"]})


def fit(df):
    """Least squares on hourly differences; rows with any NaN (incl. the next T) dropped.
    Returns (a, b, c), their standard errors, and the number of rows used."""
    y = df["T"].shift(-1) - df["T"]
    X = np.column_stack([df["t_out"] - df["T"], df["Q"], df["solar"]])
    ok = y.notna().values & ~np.isnan(X).any(axis=1)
    X, y = X[ok], y.values[ok]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = ((y - X @ coef) ** 2).sum() / (len(y) - 3)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))   # assumes iid residuals: too small if they are autocorrelated
    return coef, se, int(ok.sum())


def params(a, b, c):
    C = DT / b
    return {"R": b / a, "C": C, "A_SOL": 1000 * c * C / DT}


def coefs(p):
    return DT / (p["R"] * p["C"]), DT / p["C"], p["A_SOL"] * DT / (1000 * p["C"])


def simulate(df, a, b, c):
    T = np.empty(len(df))
    T[0] = df["T"].iloc[0]
    for k in range(len(df) - 1):
        T[k + 1] = T[k] + a * (df["t_out"].iloc[k] - T[k]) + b * df["Q"].iloc[k] + c * df["solar"].iloc[k]
    return T


def errors(meas, pred):
    e = pred - meas
    return float(np.sqrt(np.mean(e ** 2))), float(np.abs(e).max())


def main():
    df = load()
    train, val = df[df.index < SPLIT], df[df.index >= SPLIT]
    assert not val.isna().any().any(), "gap in the validation period"
    coef, se, n = fit(train)
    fitted = params(*coef)
    val_fit, val_ass = simulate(val, *coef), simulate(val, *coefs(ASSUMED))
    rmse_f, max_f = errors(val["T"].values, val_fit)
    rmse_a, max_a = errors(val["T"].values, val_ass)
    # Open-loop on the training part after the power cut too (a fit-quality check, not validation).
    tr2 = train[train.index > train["T"].isna()[::-1].idxmax()]
    rmse_tr, _ = errors(tr2["T"].values, simulate(tr2, *coef))

    rows = [("R_K_per_kW", fitted["R"]), ("C_kWh_per_K", fitted["C"]), ("A_SOL_m2", fitted["A_SOL"]),
            ("tau_h", fitted["R"] * fitted["C"]), ("UA_W_per_K", 1000 / fitted["R"]),
            ("a", coef[0]), ("b", coef[1]), ("c", coef[2]),
            ("se_a", se[0]), ("se_b", se[1]), ("se_c", se[2]), ("n_train_hours", n),
            ("train_start", train.index[0]), ("train_end", train.index[-1]),
            ("val_start", val.index[0]), ("val_end", val.index[-1]), ("n_val_hours", len(val)),
            ("val_rmse_C_fitted", rmse_f), ("val_max_err_C_fitted", max_f),
            ("val_rmse_C_assumed", rmse_a), ("val_max_err_C_assumed", max_a),
            ("train_openloop_rmse_C_fitted_after_gap", rmse_tr),
            ("R_assumed", ASSUMED["R"]), ("C_assumed", ASSUMED["C"]), ("A_SOL_assumed", ASSUMED["A_SOL"]),
            ("tau_h_assumed", ASSUMED["R"] * ASSUMED["C"])]
    with open(OUT / "calibration_zeb.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "value"])
        w.writerows(rows)
    for k, v in rows:
        print(f"{k:42s} {v}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(val.index, val["T"], color="#2b2b2b", lw=2, label="measured (extract air)")
    ax.plot(val.index, val_fit, color="#2a78d6", lw=2, label=f"fitted 1R1C, RMSE {rmse_f:.2f} C")
    ax.plot(val.index, val_ass, color="#eb6834", lw=2, label=f"assumed house, RMSE {rmse_a:.2f} C")
    ax.set_ylabel("indoor temperature (C)")
    ax.set_title("ZEB Living Lab, held-out E-MPC period (1-13 Mar 2019): open-loop simulation "
                 "with measured Tout, heat and solar", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_zeb.png", dpi=150)


if __name__ == "__main__":
    main()
