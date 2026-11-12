"""Settings -> comfort bands, the 24 h MPC (a linear program), two baselines, and
the hour-by-hour simulation loop.

The MPC plans electrical heat-pump power P_t for the next H hours:

    min  sum_t P_t*DT*(price_t + lam*co2_fc_t)  +  M * sum_t s_t
    s.t. T_min_t - s_t <= T_t <= T_max_t + s_t,   0 <= P_t <= P_MAX,   s_t >= 0
         T_{t+1} = T_t - (T_t - T_out_t)*DT/(RC) + COP(T_out_t)*P_t*DT/C + solar term

COP depends only on the (forecast) outdoor temperature, never on P or T, so
every constraint stays linear in P. We use the actual weather as a perfect
forecast (stated in the README). Only hour 1 of the plan is applied; then the
house moves on with actual data and the LP is solved again (receding horizon).
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from house import A_SOL, C, DT, P_MAX, R, cop, step

H = 24            # planning horizon, hours
M = 100.0         # kr per (deg C * h) of band violation: far above any saving, so slack is a last resort
T0 = 21.0         # indoor temperature at the start of the winter
BASE_BAND = (20.0, 22.0)   # default occupied comfort band (assumption: 21 +- 1 C)
AWAY_MIN = 16.0   # default lower bound while away, if the resident gave none (assumption)
# Priority -> (lam in kr/kg CO2, extra C on the band's lower edge). Assumptions:
# 0.75 = 2030 Danish CO2 tax, non-ETS (750 DKK/t). 5 lets CO2 outweigh price swings.
PRIORITY = {None: (0.75, 0.0), "cost": (0.0, 0.0), "co2": (5.0, 0.0), "comfort": (0.75, 0.5)}


def bands(index, msgs):
    """msgs: (sent_time, settings) in time order. Band and priority changes apply
    from sent_time on; an absence applies to its own window.
    t_min/t_max: what the MPC MUST keep. t_home: t_min if nobody left.
    occ_min/occ_max: the occupied band, for scoring."""
    n = len(index)
    occ_min, occ_max = np.full(n, BASE_BAND[0]), np.full(n, BASE_BAND[1])
    lam, bump = np.full(n, PRIORITY[None][0]), np.zeros(n)
    away = np.full(n, np.nan)            # NaN = home; otherwise the away lower bound
    hours, days = index.hour.values, index.day_of_week.values
    for sent, s in msgs:
        later = index >= pd.Timestamp(sent, tz=index.tz)
        if s.get("t_min") is not None:
            occ_min[later] = s["t_min"]
        if s.get("t_max") is not None:
            occ_max[later] = s["t_max"]
        if s.get("priority") is not None:
            lam[later], bump[later] = PRIORITY[s["priority"]]
        for a in s.get("absences", []):
            win = (index >= pd.Timestamp(a["start"], tz=index.tz)) & (index < pd.Timestamp(a["end"], tz=index.tz))
            away[win] = a.get("min_temp") or AWAY_MIN
        for w in s.get("weekly_absences", []):
            win = later & np.isin(days, w["weekdays"]) & (hours >= w["start_hour"]) & (hours < w["end_hour"])
            away[win] = w.get("min_temp") or AWAY_MIN
    occupied = np.isnan(away)
    t_home = occ_min + bump
    return {"t_min": np.where(occupied, t_home, away), "t_home": t_home, "t_max": occ_max,
            "lam": lam, "occupied": occupied,
            "occ_min": occ_min, "occ_max": occ_max}


def predict(T0, t_out, solar):
    """Write the dynamics as T_{1..H} = G @ P + f.
    One step is T_{k+1} = a*T_k + b_k*P_k + d_k, so unrolling gives
    T_{k+1} = a^(k+1)*T0 + sum_{j<=k} a^(k-j)*(b_j*P_j + d_j)."""
    a = 1 - DT / (R * C)
    b = cop(t_out) * DT / C
    d = np.asarray(t_out) * DT / (R * C) + A_SOL * np.asarray(solar) / 1000 * DT / C
    k = np.arange(len(b))
    powers = np.tril(a ** np.maximum(np.subtract.outer(k, k), 0))   # a^(k-j) for j <= k, else 0
    G = powers * b                      # column j scaled by b_j
    f = a ** (k + 1) * T0 + powers @ d
    return G, f


def solve(T0, t_out, solar, price, co2_fc, t_min, t_max, lam):
    """Decision x = [P_0..P_{H-1}, s_1..s_H]."""
    n = len(t_out)
    G, f = predict(T0, t_out, solar)
    cost = np.concatenate([DT * (np.asarray(price) + np.asarray(lam) * np.asarray(co2_fc)), np.full(n, M)])
    I = np.eye(n)
    A_ub = np.block([[G, -I],      #  G P + f - s <= t_max
                     [-G, -I]])    # -G P - f - s <= -t_min
    b_ub = np.concatenate([t_max - f, f - t_min])
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=[(0, P_MAX)] * n + [(0, None)] * n, method="highs")
    assert res.status == 0, res.message
    P, s = res.x[:n], res.x[n:]
    return P, G @ P + f, s


def horizon(df, b, t, use_absences=True):
    """Weather, price and CO2 are for hours t..t+n-1, when P acts. The band is for
    states T_1..T_n, the start of hours t+1..t+n, so it is shifted by one."""
    n = min(H, len(df) - t)
    sl = slice(t, t + n)
    nxt = np.minimum(np.arange(t + 1, t + 1 + n), len(df) - 1)
    lo = b["t_min"] if use_absences else b["t_home"]
    return dict(t_out=df["t_out"].values[sl], solar=df["solar"].values[sl], price=df["price"].values[sl],
                co2_fc=df["co2_fc"].values[sl], t_min=lo[nxt], t_max=b["t_max"][nxt], lam=b["lam"][sl])


def what_if(T0, hz, **change):
    """Alternative minus current plan. cold_dh is measured against the CURRENT band.
    The alternative MUST end at least as warm. Otherwise it coasts down ~1 C, and
    the heat bought back tomorrow (~10 kWh) counts as a saving."""
    out = {}
    P, T, _ = solve(T0, **hz)
    alt = {**hz, **change}
    alt["t_min"] = np.append(alt["t_min"][:-1], max(alt["t_min"][-1], T[-1]))
    for name, h in [("now", hz), ("alt", alt)]:
        P, T, _ = solve(T0, **h)
        out[name] = {"kwh": P.sum() * DT, "kr": (P * h["price"]).sum() * DT,
                     "kg": (P * h["co2_fc"]).sum() * DT,
                     "cold_dh": np.clip(hz["t_min"] - T, 0, None).sum() * DT, "min_T": T.min()}
    diff = {k: out["alt"][k] - out["now"][k] for k in ["kwh", "kr", "kg", "cold_dh"]}
    return out, diff


# A controller maps (hour t, indoor T, bands b) to power P in kW.

def p_to_reach(T, target, t_out, solar):
    """A perfect one-step thermostat on the same house model."""
    free = step(T, 0.0, t_out, solar)
    return float(np.clip((target - free) * C / (cop(t_out) * DT), 0, P_MAX))


def fixed(df, target=21.0):
    return lambda t, T, b: p_to_reach(T, target, df["t_out"].values[t], df["solar"].values[t])


def rule_based(df):
    """Ignores absences: it only sees the occupied band."""
    day = df.index.date
    cheap = (df["price"].groupby(day).rank(pct=True) <= 0.25).values

    def ctrl(t, T, b):
        target = b["occ_max"][t] if cheap[t] else b["occ_min"][t]
        return p_to_reach(T, target, df["t_out"].values[t], df["solar"].values[t])
    return ctrl


def mpc(df, use_absences=True):
    def ctrl(t, T, b):
        P, _, _ = solve(T, **horizon(df, b, t, use_absences))
        return P[0]                      # apply hour 1 only
    return ctrl


def simulate(df, ctrl, versions, sent_times):
    """versions[k] = bands() from the first k messages. At hour t the controller
    sees only messages sent at or before t."""
    n = len(df)
    known = np.searchsorted(np.asarray(sent_times, dtype="datetime64[ns]"),
                            df.index.tz_localize(None).values, side="right")
    T, Ts, Ps = T0, np.zeros(n), np.zeros(n)
    for t in range(n):
        P = ctrl(t, T, versions[known[t]])
        Ts[t], Ps[t] = T, P
        T = step(T, P, df["t_out"].values[t], df["solar"].values[t])
    return Ts, Ps


def score(df, Ts, Ps, truth):
    """Cost and CO2 use ACTUAL series. Comfort counts truly occupied hours only."""
    occ = truth["occupied"]
    cold = np.clip(truth["occ_min"] - Ts, 0, None)
    hot = np.clip(Ts - truth["occ_max"], 0, None)
    return {"kwh": Ps.sum() * DT, "kr": (Ps * df["price"].values).sum() * DT,
            "kg_co2": (Ps * df["co2"].values).sum() * DT,
            "violation_dh": float(((cold + hot) * occ).sum() * DT),
            "mean_T_occupied": float(Ts[occ].mean())}
