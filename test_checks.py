"""Spec checks. Synthetic weather/price so they run fast and offline
(except test 4, which reads requests.jsonl)."""
import json

import numpy as np
import pandas as pd

import explain
import llm
import mpc
from house import P_MAX, cop, step
from safety import validate, violations


def synthetic(hours=48):
    t = np.arange(hours)
    idx = pd.date_range("2026-01-12", periods=hours, freq="h", tz="Europe/Copenhagen")
    return pd.DataFrame({"t_out": -2 + 3 * np.sin(2 * np.pi * (t - 9) / 24),
                         "solar": np.clip(200 * np.sin(2 * np.pi * (t - 8) / 24), 0, None),
                         "price": 0.6 + 0.4 * np.sin(2 * np.pi * (t - 12) / 12) ** 2,
                         "co2": 0.10 + 0.03 * np.cos(2 * np.pi * t / 24),
                         "co2_fc": 0.10 + 0.03 * np.cos(2 * np.pi * t / 24)}, index=idx)


def test_1_house_steady_state_matches_hand_calculation():
    # Hand calculation: COP(0 C) = 2 + 0.125 * 5 = 2.625. With P = 1 kW the heat
    # input is 2.625 kW; steady state when loss (T - T_out)/R equals it:
    # T = 0 + R * 2.625 = (1 / 0.15) * 2.625 = 17.5 C.
    assert np.isclose(cop(0), 2.625)
    assert np.isclose(step(17.5, 1.0, 0.0, 0.0), 17.5)
    assert np.isclose(cop(-5), 2.0) and np.isclose(cop(7), 3.5)
    assert cop(-20) == 1.5 and cop(30) == 4.0           # clamped


def test_2_lp_respects_bounds_and_slack_is_zero_when_feasible():
    df = synthetic(24)
    tmin, tmax = np.full(24, 20.0), np.full(24, 22.0)
    P, T, s = mpc.solve(21.0, df["t_out"].values, df["solar"].values, df["price"].values,
                        df["co2_fc"].values, tmin, tmax, np.full(24, 0.75))
    assert P.min() >= -1e-9 and P.max() <= P_MAX + 1e-9
    assert s.max() < 1e-6
    assert T.min() >= 20 - 1e-6 and T.max() <= 22 + 1e-6
    # The LP's prediction MUST equal stepping the house model with the plan.
    x = 21.0
    for k in range(24):
        x = step(x, P[k], df["t_out"].values[k], df["solar"].values[k])
        assert np.isclose(x, T[k])


def test_3_mpc_no_dearer_than_fixed_21_at_equal_or_better_comfort():
    df = synthetic(48)
    vs = [mpc.bands(df.index, [])]                # nobody leaves, band 20-22
    res = {}
    for name, ctrl in [("fixed", mpc.fixed(df)), ("mpc", mpc.mpc(df))]:
        Ts, Ps = mpc.simulate(df, ctrl, vs, [])
        res[name] = mpc.score(df.iloc[:24], Ts[:24], Ps[:24],          # score the first day
                              {k: v[:24] for k, v in vs[0].items()})
    assert res["mpc"]["kr"] <= res["fixed"]["kr"] + 1e-9
    assert res["mpc"]["violation_dh"] <= res["fixed"]["violation_dh"] + 1e-6


def test_4_validator_clamps_every_unsafe_gold_example():
    unsafe = [r for r in llm.load_requests() if r["category"] == "unsafe"]
    assert len(unsafe) >= 10
    for r in unsafe:
        g = llm.gold(r)
        assert violations(g), f"gold of request {r['id']} is not actually unsafe"
        clean, reasons = validate(g)
        assert violations(clean) == [], (r["id"], violations(clean))
        assert reasons, r["id"]                    # every change comes with a reason
    out, reasons = validate(dict(llm.gold({"gold": {"out_of_scope": True}}), t_min=30))
    assert out["t_min"] is None and reasons[0].startswith("rejected")


def test_5_faithfulness_check_catches_planted_wrong_number():
    plan = {"kwh": 25.3, "kr": 17.84, "kg": 2.71, "min_T": 20.0}
    diff = {"kwh": -1.2, "kr": -0.93, "kg": -0.14, "cold_dh": 11.6}
    f = explain.facts(plan, diff)
    good, _ = explain.explain(f, stub_id=0)
    bad, _ = explain.explain(f, stub_id=2)          # stub plants kr saving + 1.0
    assert explain.check(good, f)[0]
    ok, wrong = explain.check(bad, f)
    assert not ok and wrong == ["1.9 kr"]
    # Hand-written: Danish decimal comma accepted, "CO2" not read as a number, wrong kWh caught.
    assert explain.check("Planen koster 17,8 kr og 2,7 kg CO2.", f)[0]
    assert explain.check("It uses 30.0 kWh.", f) == (False, ["30.0 kWh"])


def test_6_llm_cache_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache.json")
    monkeypatch.setenv("OLLAMA_MODEL", "some-model")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")   # nothing listens: a miss would fail
    msgs = llm.prompt("Lower it.", "2026-02-24T18:00")
    llm.cache_put(llm.cache_key("some-model", msgs, llm.SCHEMA), {"content": '{"t_min": 19}', "latency_s": 1.5})
    assert llm.chat(msgs, llm.SCHEMA) == ('{"t_min": 19}', 1.5)
    s, latency, model = llm.parse({"text": "Lower it.", "sent": "2026-02-24T18:00"})
    assert s["t_min"] == 19 and s["absences"] == [] and latency == 1.5 and model == "some-model"


def test_7_deepseek_request_body_offline():
    # Builds the request only; nothing is sent. The key is a fake.
    msgs = llm.prompt("Away all weekend.", "2026-02-24T18:00")
    headers, body = llm.deepseek_request(msgs, llm.SCHEMA, "deepseek-flash", "fake-key")
    assert headers == {"Authorization": "Bearer fake-key"}
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0 and body["stream"] is False
    system = body["messages"][0]["content"]
    assert json.dumps(llm.SCHEMA, separators=(",", ":")) in system and "json" in system
    assert msgs[0]["content"] == llm.SYSTEM          # the caller's messages (the cache key) are untouched
    _, plain = llm.deepseek_request(msgs, None, "deepseek-flash", "fake-key")   # explanations: no JSON mode
    assert "response_format" not in plain and plain["messages"] == msgs


def test_8_calibration_recovers_known_R_and_C():
    # Synthetic data from house.step with known R, C, A_SOL; small process noise on each step.
    import calibrate
    import house
    rng = np.random.default_rng(0)
    n = 500
    t = np.arange(n)
    t_out = 2 + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, n)
    P = rng.choice([0.0, 1.0, 2.5], n)                  # PRBS-like excitation, kW electrical
    solar = np.clip(300 * np.sin(2 * np.pi * (t - 6) / 24), 0, None)
    T = np.empty(n)
    T[0] = 21.0
    for k in range(n - 1):
        T[k + 1] = step(T[k], P[k], t_out[k], solar[k]) + rng.normal(0, 0.01)
    df = pd.DataFrame({"T": T, "t_out": t_out, "Q": cop(t_out) * P, "solar": solar})
    coef, _, _ = calibrate.fit(df)
    p = calibrate.params(*coef)
    assert abs(p["R"] / house.R - 1) < 0.03 and abs(p["C"] / house.C - 1) < 0.03
