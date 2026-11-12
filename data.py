"""Fetch, cache and align the winter inputs for one house in Aarhus (DK1).

`python data.py` MUST run once (online). After that `load()` works offline from
the CSVs in data/.

Sources (free, no key):
- Energi Data Service (Energinet): CO2Emis (realised g CO2/kWh), CO2EmisProg
  (prognosis), DayAheadPrices (DKK/MWh, 15 min, from 2025-10-01). Before
  2025-10-01 prices live in the retired hourly dataset Elspotprices.
- Open-Meteo archive: hourly 2 m temperature and shortwave radiation at Aarhus.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
API = "https://api.energidataservice.dk/dataset/"
METEO = "https://archive-api.open-meteo.com/v1/archive"
REGION = "DK1"                       # Aarhus is in DK1
LAT, LON = 56.15, 10.21              # Aarhus
TZ = "Europe/Copenhagen"
# Winter window in Danish local time (end exclusive). The EDS API reads
# start/end as Danish local time, so these strings go to it unchanged.
START, END = "2025-12-01T00:00", "2026-03-01T00:00"
# DayAheadPrices starts here; earlier prices are only in Elspotprices.
DAYAHEAD_FROM = "2025-10-01T00:00"


def get(url, params):
    import requests

    for attempt in range(6):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()
        print("rate limited, waiting 60 s ...")
        time.sleep(60)
    r.raise_for_status()


def eds(dataset, start, end, time_col):
    params = {"start": start, "end": end, "filter": json.dumps({"PriceArea": [REGION]}),
              "sort": f"{time_col} asc", "limit": 0}
    df = pd.DataFrame(get(API + dataset, params)["records"])
    if df.empty:
        raise ValueError(f"{dataset}: no records returned for {start}..{end}")
    return df


def fetch_prices(start, end):
    parts = []
    if start < DAYAHEAD_FROM:
        old = eds("Elspotprices", start, min(end, DAYAHEAD_FROM), "HourUTC")
        parts.append(pd.DataFrame({"TimeUTC": old["HourUTC"], "price_dkk_mwh": old["SpotPriceDKK"],
                                   "dataset": "Elspotprices"}))
    if end > DAYAHEAD_FROM:
        new = eds("DayAheadPrices", max(start, DAYAHEAD_FROM), end, "TimeUTC")
        parts.append(pd.DataFrame({"TimeUTC": new["TimeUTC"], "price_dkk_mwh": new["DayAheadPriceDKK"],
                                   "dataset": "DayAheadPrices"}))
    return pd.concat(parts, ignore_index=True)


def fetch_weather(start, end):
    """Open-Meteo takes whole UTC dates, so ask one day wider each side. load() trims."""
    s = (pd.Timestamp(start) - pd.Timedelta(days=1)).date()
    e = (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
    j = get(METEO, {"latitude": LAT, "longitude": LON, "start_date": str(s), "end_date": str(e),
                    "hourly": "temperature_2m,shortwave_radiation", "timezone": "UTC"})
    return pd.DataFrame({"TimeUTC": j["hourly"]["time"], "t_out": j["hourly"]["temperature_2m"],
                         "solar_w_m2": j["hourly"]["shortwave_radiation"]})


def download():
    DATA_DIR.mkdir(exist_ok=True)
    for name, ds in [("co2_actual", "CO2Emis"), ("co2_forecast", "CO2EmisProg")]:
        eds(ds, START, END, "Minutes5UTC")[["Minutes5UTC", "CO2Emission"]].to_csv(
            DATA_DIR / f"{name}.csv", index=False)
    fetch_prices(START, END).to_csv(DATA_DIR / "prices.csv", index=False)
    fetch_weather(START, END).to_csv(DATA_DIR / "weather.csv", index=False)
    print(f"fetched {START} .. {END} (Danish local time)")


def to_hourly(df, time_col, cols):
    df = df.copy()
    df["t"] = pd.to_datetime(df[time_col]).dt.floor("h").dt.tz_localize("UTC")
    return df.groupby("t")[cols].mean()


def load(verbose=False):
    """Return (df, info). df: hourly, index = local time (Europe/Copenhagen),
    columns price (kr/kWh), co2 (actual kg/kWh), co2_fc (forecast kg/kWh),
    t_out (deg C), solar (W/m2). Missing hours are linearly interpolated and counted."""
    raw_co2 = pd.read_csv(DATA_DIR / "co2_actual.csv")
    raw_fc = pd.read_csv(DATA_DIR / "co2_forecast.csv")
    co2 = to_hourly(raw_co2, "Minutes5UTC", ["CO2Emission"])
    fc = to_hourly(raw_fc, "Minutes5UTC", ["CO2Emission"])
    raw_prices = pd.read_csv(DATA_DIR / "prices.csv")
    prices = to_hourly(raw_prices, "TimeUTC", ["price_dkk_mwh"])
    wx = to_hourly(pd.read_csv(DATA_DIR / "weather.csv"), "TimeUTC", ["t_out", "solar_w_m2"])

    df = pd.DataFrame({
        "price": prices["price_dkk_mwh"] / 1000,    # DKK/MWh -> kr/kWh
        "co2": co2["CO2Emission"] / 1000,           # g/kWh -> kg/kWh
        "co2_fc": fc["CO2Emission"] / 1000,
        "t_out": wx["t_out"], "solar": wx["solar_w_m2"],
    })
    full = pd.date_range(pd.Timestamp(START, tz=TZ), pd.Timestamp(END, tz=TZ),
                         freq="h", inclusive="left").tz_convert("UTC")
    df = df.reindex(full)
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    df = df.interpolate(limit_direction="both")
    df.index = df.index.tz_convert(TZ)

    info = {
        "start_local": str(df.index[0]), "end_local": str(df.index[-1]), "rows_hourly": len(df),
        "hours_interpolated": missing,
        # Caveat: an hourly mean is kept even if some of its 5-/15-min values are missing.
        "raw_records": {"co2_actual_5min": len(raw_co2), "co2_forecast_5min": len(raw_fc),
                        "prices_15min": len(raw_prices), "complete_5min": 12 * len(df),
                        "complete_15min": 4 * len(df)},
        "price_datasets": sorted(raw_prices["dataset"].unique().tolist()),
        "mean_price_kr_kwh": float(df["price"].mean()), "min_price_kr_kwh": float(df["price"].min()),
        "max_price_kr_kwh": float(df["price"].max()),
        "mean_co2_actual_g_kwh": float(1000 * df["co2"].mean()),
        "mean_co2_forecast_g_kwh": float(1000 * df["co2_fc"].mean()),
        "co2_forecast_mae_g_kwh": float(1000 * (df["co2_fc"] - df["co2"]).abs().mean()),
        "mean_t_out_c": float(df["t_out"].mean()), "min_t_out_c": float(df["t_out"].min()),
        "max_t_out_c": float(df["t_out"].max()),
        "mean_solar_w_m2": float(df["solar"].mean()),
        "corr_price_co2": float(df["price"].corr(df["co2"])),
    }
    if verbose:
        print(f"range (local): {info['start_local']} .. {info['end_local']}, hourly rows: {len(df)}")
        print("hours missing before interpolation:", missing)
        print("raw records:", info["raw_records"])
        print("price datasets used:", info["price_datasets"])
        for c, unit, k in [("price", "kr/kWh", 1), ("co2", "g/kWh", 1000), ("co2_fc", "g/kWh", 1000),
                           ("t_out", "C", 1), ("solar", "W/m2", 1)]:
            s = k * df[c]
            print(f"{c:7s} min/mean/max {s.min():8.2f} / {s.mean():8.2f} / {s.max():8.2f}  {unit}")
        print(f"CO2 forecast MAE: {info['co2_forecast_mae_g_kwh']:.1f} g/kWh, "
              f"corr(price, CO2): {info['corr_price_co2']:.3f}")
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "data_info.json").write_text(json.dumps(info, indent=2))
    return df, info


if __name__ == "__main__":
    download()
    load(verbose=True)
