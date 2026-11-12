"""1R1C model of a Danish single-family house heated by an air-to-water heat pump.

One thermal resistance R (house -> outdoors) and one heat capacity C (the whole
house as one lump). All numbers are ASSUMPTIONS, not measurements:

- R = 6.67 K/kW, i.e. heat loss 150 W/K: a ~150 m2 house that has had some
  renovation (about 4.8 kW loss at 22 C inside / -10 C outside).
- C = 10 kWh/K: a medium-heavy house; time constant R*C = 67 h, so with the
  heating off at 22 C inside it cools (22 - T_out)/67 K per hour: ~0.3 C/h at
  0 C outside, ~0.5 C/h at -10 C.
- COP is linear in outdoor temperature through 2.0 at -5 C and 3.5 at 7 C,
  clamped to [1.5, 4.0] (typical air-to-water datasheet shape at ~35 C flow).
- P_MAX = 4 kW electrical: enough to hold 22 C at the coldest hour (-10 C).
- A_SOL = 2 m2 effective solar aperture: low winter sun, small windows; at the
  window's peak of ~390 W/m2 that is ~0.8 kW of free heat.
"""
import csv
import os
from pathlib import Path

import numpy as np

ASSUMED = {"R": 1 / 0.15, "C": 10.0, "A_SOL": 2.0}   # K/kW, kWh/K, m2
R, C, A_SOL = ASSUMED["R"], ASSUMED["C"], ASSUMED["A_SOL"]
# HOUSE=calibrated: R, C, A_SOL fitted to the ZEB Living Lab (calibrate.py). COP and P_MAX stay assumed.
if os.environ.get("HOUSE") == "calibrated":
    with open(Path(__file__).parent / "results" / "calibration_zeb.csv") as f:
        fit = {row["name"]: row["value"] for row in csv.DictReader(f)}
    R, C, A_SOL = (float(fit[k]) for k in ("R_K_per_kW", "C_kWh_per_K", "A_SOL_m2"))
P_MAX = 4.0       # kW electrical
DT = 1.0          # h, one step = one hour
# COP line through 2.0 at -5 C and 3.5 at 7 C: slope 1.5/12 = 0.125 per K.
COP_T_REF, COP_REF, COP_SLOPE = -5.0, 2.0, 0.125
COP_LO, COP_HI = 1.5, 4.0


def cop(t_out):
    return np.clip(COP_REF + COP_SLOPE * (np.asarray(t_out) - COP_T_REF), COP_LO, COP_HI)


def step(T, P, T_out, solar):
    """Explicit Euler. Stable: DT/(RC) is ~0.015-0.017, far below 1. P in kW, solar in W/m2."""
    loss = (T - T_out) * DT / (R * C)            # heat leaking out, in K
    heat = cop(T_out) * P * DT / C               # heat-pump heat, in K
    sun = A_SOL * solar / 1000 * DT / C          # W/m2 * m2 -> kW, then K
    return T - loss + heat + sun
