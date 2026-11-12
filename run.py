"""Simulate one Aarhus winter for 3 households and write results/ (CSVs + plots).

    python run.py --skip-llm   experiments 1-3 with gold (ground-truth) settings only
    python run.py --stub       everything; LLM replaced by the labelled STUB (no Ollama needed)
    python run.py              everything with the real model in $OLLAMA_MODEL

Works offline from data/; `python data.py` MUST have run once first.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import explain
import llm
import mpc
from safety import validate, violations

RESULTS = data.RESULTS
SUFFIX = "_calibrated" if os.environ.get("HOUSE") == "calibrated" else ""   # keeps assumed-house files untouched
HOUSEHOLDS = {"commuter": "commuter couple", "homeworker": "home worker", "family": "family with kids"}
BAND_HALF_WIDTHS = [0.5, 1.0, 1.5, 2.0]   # exp 2: band = 21 +- w
P_ACCEPT = [0.0, 0.25, 0.5, 0.75, 1.0]    # exp 3: assumed chance a resident accepts "1 C lower" today
OFFER_SEED = 0                            # exp 3 acceptance draws
EXPLAIN_EVERY_DAYS = 10                   # exp 4: one explanation per household every 10 days, at 17:00
FIELDS = ["absences", "weekly_absences", "t_min", "t_max", "priority", "needs_clarification", "out_of_scope"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # fixed order: one per controller


def household_msgs(reqs, hh, settings):
    mine = sorted((r for r in reqs if r["household"] == hh), key=lambda r: r["sent"])
    return [(r["sent"], validate(settings(r))[0]) for r in mine]


def versions(df, msgs):
    return [mpc.bands(df.index, msgs[:k]) for k in range(len(msgs) + 1)], [s for s, _ in msgs]


def run(df, ctrl, msgs, truth, shift=None):
    """shift: per-hour C taken off the band's lower edge (exp 3)."""
    vs, sent = versions(df, msgs)
    if shift is not None:
        vs = [dict(v, t_min=v["t_min"] - shift, t_home=v["t_home"] - shift) for v in vs]
    Ts, Ps = mpc.simulate(df, ctrl, vs, sent)
    return mpc.score(df, Ts, Ps, truth), Ts, Ps


def same(field, a, b):
    try:
        if field == "absences":
            key = lambda x: (pd.Timestamp(x["start"]), pd.Timestamp(x["end"]), x.get("min_temp"))
            return sorted(map(key, a)) == sorted(map(key, b))
        if field == "weekly_absences":
            key = lambda x: (sorted(x["weekdays"]), x["start_hour"], x["end_hour"], x.get("min_temp"))
            return sorted(map(key, a)) == sorted(map(key, b))
        if field in ("t_min", "t_max") and a is not None and b is not None:
            return abs(float(a) - float(b)) < 1e-6
        return a == b
    except (KeyError, TypeError, ValueError):
        return False


def bar_panels(df_, metrics, titles, path, suptitle):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8))
    ctrls = list(dict.fromkeys(df_["controller"]))
    x = np.arange(len(HOUSEHOLDS))
    w = 0.8 / len(ctrls)
    for ax, m, title in zip(axes, metrics, titles):
        for i, c in enumerate(ctrls):
            vals = df_[df_["controller"] == c].set_index("household").loc[list(HOUSEHOLDS), m]
            ax.bar(x + i * w, vals, w * 0.92, color=COLORS[i], label=c)
        ax.set_xticks(x + w * (len(ctrls) - 1) / 2, [HOUSEHOLDS[h] for h in HOUSEHOLDS], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def exp1_occupancy(df, reqs, parsed=None):
    rows, traces = [], {}
    for hh in HOUSEHOLDS:
        gold_msgs = household_msgs(reqs, hh, llm.gold)
        truth = versions(df, gold_msgs)[0][-1]
        ctrls = {"fixed_21C": (mpc.fixed(df), gold_msgs),
                 "rule_cheap25": (mpc.rule_based(df), gold_msgs),
                 "mpc_no_occupancy": (mpc.mpc(df, use_absences=False), gold_msgs),
                 "mpc_ground_truth": (mpc.mpc(df), gold_msgs)}
        if parsed is not None:
            ctrls["mpc_llm"] = (mpc.mpc(df), household_msgs(reqs, hh, lambda r: parsed[r["id"]]))
        for name, (ctrl, msgs) in ctrls.items():
            sc, Ts, Ps = run(df, ctrl, msgs, truth)
            rows.append({"household": hh, "controller": name, **sc})
            traces[hh, name] = Ts
            print(f"  exp1 {hh:10s} {name:17s} kWh {sc['kwh']:7.0f}  kr {sc['kr']:7.0f}  "
                  f"kg {sc['kg_co2']:6.0f}  violation {sc['violation_dh']:6.1f} C*h")
    return pd.DataFrame(rows), traces


def exp2_band_sweep(df, reqs):
    """Band forced to 21 +- w; household band messages are dropped."""
    rows = []
    for hh in HOUSEHOLDS:
        for w in BAND_HALF_WIDTHS:
            msgs = [(s, dict(x, t_min=None, t_max=None)) for s, x in household_msgs(reqs, hh, llm.gold)]
            msgs = [(data.START, {"t_min": 21 - w, "t_max": 21 + w})] + msgs
            truth = versions(df, msgs)[0][-1]
            sc, _, _ = run(df, mpc.mpc(df), msgs, truth)
            rows.append({"household": hh, "half_width_C": w, **sc})
            print(f"  exp2 {hh:10s} +-{w:.1f} C  kr {sc['kr']:7.0f}  kg {sc['kg_co2']:6.0f}")
    return pd.DataFrame(rows)


def exp3_sufficiency(df, reqs):
    """One uniform draw per day, shared by every p, so a larger p accepts a
    superset of days. Comfort is scored against the real (unshifted) band."""
    rows = []
    days = pd.Index(df.index.date)
    n_days = days.nunique()
    for i, hh in enumerate(HOUSEHOLDS):
        gold_msgs = household_msgs(reqs, hh, llm.gold)
        vs, sent = versions(df, gold_msgs)
        truth = vs[-1]
        u = np.random.default_rng(OFFER_SEED + i).random(n_days)
        day_no = days.factorize()[0]
        base = None
        for p in P_ACCEPT:
            shift = (u < p)[day_no].astype(float)          # 1 C lower on accepted days
            sc, Ts, _ = run(df, mpc.mpc(df), gold_msgs, truth, shift)
            if p == 0:
                base, base_T = sc, Ts
                # What the controller would quote at each midnight (24 h what-if, forecast CO2).
                known = np.searchsorted(np.asarray(sent, dtype="datetime64[ns]"),
                                        df.index.tz_localize(None).values, side="right")
                quotes = []
                for t in range(0, len(df), 24):
                    hz = mpc.horizon(df, vs[known[t]], t)
                    _, d = mpc.what_if(base_T[t], hz, t_min=hz["t_min"] - 1)
                    quotes.append(d)
                quoted_kr = -sum(q["kr"] for q in quotes)
                quoted_kg = -sum(q["kg"] for q in quotes)
            rows.append({"household": hh, "p_accept": p, "accepted_days": int((u < p).sum()),
                         "saving_kr": base["kr"] - sc["kr"], "saving_kg_co2": base["kg_co2"] - sc["kg_co2"],
                         "extra_violation_dh": sc["violation_dh"] - base["violation_dh"],
                         # sum of what_if quotes x p: the "expected" saving the offers promised
                         "quoted_expected_kr": p * quoted_kr, "quoted_expected_kg_co2": p * quoted_kg, **sc})
            print(f"  exp3 {hh:10s} p={p:.2f}  saving kr {rows[-1]['saving_kr']:6.1f}  "
                  f"kg {rows[-1]['saving_kg_co2']:5.1f}  quoted kr {p * quoted_kr:6.1f}")
    return pd.DataFrame(rows)


def exp4_llm(df, reqs, stub, traces):
    rows, parsed = [], {}
    for r in reqs:
        s, latency, model = llm.parse(r, stub)
        parsed[r["id"]] = s
        g = llm.gold(r)
        clean, reasons = validate(s)
        row = {"id": r["id"], "category": r["category"], "lang": r["lang"], "model": model,
               "latency_s": latency, "parse_error": bool(s.get("parse_error")),
               "unsafe_raw": len(violations(s)) > 0, "unsafe_after_validator": len(violations(clean)) > 0,
               "validator_reasons": " | ".join(reasons)}
        for f in FIELDS:
            row[f] = same(f, s.get(f), g[f])
        row["all_fields"] = all(row[f] for f in FIELDS)
        rows.append(row)
    req_df = pd.DataFrame(rows)

    # Explanations: sample decision points on the ground-truth MPC run.
    ex_rows = []
    for hh in HOUSEHOLDS:
        vs, sent = versions(df, household_msgs(reqs, hh, llm.gold))
        Ts = traces[hh, "mpc_ground_truth"]
        for t in range(17, len(df) - 24, 24 * EXPLAIN_EVERY_DAYS):
            k = int(np.searchsorted(np.asarray(sent, dtype="datetime64[ns]"),
                                    np.datetime64(df.index[t].tz_localize(None)), side="right"))
            hz = mpc.horizon(df, vs[k], t)
            out, diff = mpc.what_if(Ts[t], hz, t_min=hz["t_min"] - 1)
            f = explain.facts(out["now"], diff)
            text, latency = explain.explain(f, stub_id=len(ex_rows) if stub else None)
            ok, bad = explain.check(text, f)
            ex_rows.append({"household": hh, "time": str(df.index[t]), "faithful": ok, "wrong_numbers": "; ".join(bad),
                            "latency_s": latency, "text": text})
    ex_df = pd.DataFrame(ex_rows)

    clar = req_df[req_df["category"] == "ambiguous"]
    summary = {f"acc_{f}": req_df[f].mean() for f in FIELDS + ["all_fields"]}
    summary.update({
        "n_requests": len(req_df),
        "clarification_recall": clar["needs_clarification"].mean(),
        "unsafe_passed_llm_alone": int(req_df["unsafe_raw"].sum()),
        "unsafe_passed_after_validator": int(req_df["unsafe_after_validator"].sum()),
        "faithfulness_rate": ex_df["faithful"].mean(), "n_explanations": len(ex_df),
        "parse_latency_mean_s": req_df["latency_s"].mean(), "parse_latency_p95_s": req_df["latency_s"].quantile(0.95),
        "explain_latency_mean_s": ex_df["latency_s"].mean(),
    })
    by_cat = req_df.groupby("category")["all_fields"].mean()
    summary.update({f"acc_all_fields_{c}": v for c, v in by_cat.items()})
    return parsed, req_df, ex_df, pd.Series(summary, name="value")


def plot_trace(df, traces, truth, hh="commuter", start="2026-01-12", days=7):
    sl = slice(df.index.searchsorted(pd.Timestamp(start, tz=data.TZ)), None)
    sl = slice(sl.start, sl.start + 24 * days)
    t = df.index[sl]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.fill_between(t, truth["t_min"][sl], truth["t_max"][sl], step="post", color="#dddddd", label="band MPC must keep")
    for i, c in enumerate(["fixed_21C", "rule_cheap25", "mpc_no_occupancy", "mpc_ground_truth"]):
        a1.plot(t, traces[hh, c][sl], color=COLORS[i], lw=1.6, label=c)
    a1.set_ylabel("indoor temperature (C)")
    a1.legend(fontsize=7, frameon=False, ncol=3, loc="lower left")
    a2.step(t, df["price"].values[sl], where="post", color="#52514e", lw=1.2)
    a2.set_ylabel("day-ahead price (kr/kWh)")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.3)
    a1.set_title(f"{HOUSEHOLDS[hh]}, week of {start} (Mon-Fri 8-17 away)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / f"week_trace{SUFFIX}.png", dpi=150)
    plt.close(fig)


def plot_lines(df_, x, y, xlabel, ylabel, title, path, y2=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, hh in enumerate(HOUSEHOLDS):
        d = df_[df_["household"] == hh]
        ax.plot(d[x], d[y], marker="o", color=COLORS[i], lw=2, label=HOUSEHOLDS[hh])
        if y2:
            ax.plot(d[x], d[y2], ls="--", marker="x", color=COLORS[i], lw=1, label=f"{HOUSEHOLDS[hh]} (quoted)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true", help="experiments 1-3 with gold settings only")
    ap.add_argument("--stub", action="store_true", help="use the STUB instead of a model (also LLM_STUB=1)")
    args = ap.parse_args()
    stub = args.stub or os.environ.get("LLM_STUB") == "1"
    if not args.skip_llm and not stub and not llm.model_name():
        raise SystemExit("Set OLLAMA_MODEL (or LLM_BACKEND=deepseek), or run with --stub or --skip-llm.")

    RESULTS.mkdir(exist_ok=True)
    df, _ = data.load(verbose=True)
    reqs = llm.load_requests()

    print("\nExperiment 1: occupancy")
    exp1, traces = exp1_occupancy(df, reqs)
    exp1.to_csv(RESULTS / f"exp1_occupancy{SUFFIX}.csv", index=False)
    bar_panels(exp1, ["kr", "kg_co2", "violation_dh"],
               ["cost (kr, actual price)", "CO2 (kg, actual intensity)", "comfort violation (C*h, occupied)"],
               RESULTS / f"exp1_occupancy{SUFFIX}.png", "Dec 2025 - Feb 2026, Aarhus (DK1), ground-truth settings")
    truth = versions(df, household_msgs(reqs, "commuter", llm.gold))[0][-1]
    plot_trace(df, traces, truth)

    print("\nExperiment 2: comfort band sweep")
    exp2 = exp2_band_sweep(df, reqs)
    exp2.to_csv(RESULTS / f"exp2_band_sweep{SUFFIX}.csv", index=False)
    plot_lines(exp2, "half_width_C", "kr", "band half-width w (band = 21 +- w C)", "winter cost (kr)",
               "MPC cost vs comfort band width", RESULTS / f"exp2_band_sweep{SUFFIX}.png")

    print("\nExperiment 3: sufficiency offer (1 C lower)")
    exp3 = exp3_sufficiency(df, reqs)
    exp3.to_csv(RESULTS / f"exp3_sufficiency{SUFFIX}.csv", index=False)
    plot_lines(exp3, "p_accept", "saving_kr", "assumed acceptance rate p", "winter saving vs p = 0 (kr)",
               "'1 C lower today' offers: simulated (solid) vs quoted x p (dashed)",
               RESULTS / f"exp3_sufficiency{SUFFIX}.png", y2="quoted_expected_kr")

    if args.skip_llm:
        return
    tag = "STUB" if stub else re.sub(r"[^A-Za-z0-9._-]", "_", llm.model_name())
    print(f"\nExperiment 4: LLM evaluation [{tag}]" + ("  *** STUB: gold labels + planted errors, NOT a model ***" if stub else ""))
    parsed, req_df, ex_df, summary = exp4_llm(df, reqs, stub, traces)
    for d in (req_df, ex_df):
        d.insert(0, "llm_mode", tag)
    req_df.to_csv(RESULTS / f"llm_requests_{tag}{SUFFIX}.csv", index=False)
    ex_df.to_csv(RESULTS / f"llm_explanations_{tag}{SUFFIX}.csv", index=False)
    summary = summary.astype(object)
    summary["llm_mode"] = tag
    summary.to_csv(RESULTS / f"llm_eval_{tag}{SUFFIX}.csv")
    print(summary.to_string())

    print(f"\nExperiment 1 with LLM-parsed settings [{tag}]")
    exp1_llm, _ = exp1_occupancy(df, reqs, parsed)
    exp1_llm.insert(0, "llm_mode", tag)
    exp1_llm.to_csv(RESULTS / f"exp1_occupancy_with_llm_{tag}{SUFFIX}.csv", index=False)


if __name__ == "__main__":
    main()
