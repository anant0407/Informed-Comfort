"""Deterministic safety validator, applied to every Settings dict after the LLM.

The LLM's job is to say what the resident asked for; this file decides what the
house is allowed to do. Limits (assumptions, see README):
- Occupied setpoints MUST be within 16-24 C. Below 16 is a health risk; above 24 is waste.
- Away, the house MUST NOT go below 12 C (frost and mould guard).
- One absence MUST NOT exceed 21 days. Longer needs a human, not a chat message.
- Out-of-scope requests MUST change nothing. Ambiguous ones wait for an answer.
"""
import copy

import pandas as pd

OCC_LO, OCC_HI = 16.0, 24.0
AWAY_LO = 12.0
MAX_ABSENCE = pd.Timedelta(days=21)
EMPTY = {"absences": [], "weekly_absences": [], "t_min": None, "t_max": None, "priority": None,
         "needs_clarification": False, "question": None, "out_of_scope": False}


def clamp(x, lo, hi, what, reasons):
    if x is None:
        return None
    y = min(max(float(x), lo), hi)
    if y != x:
        reasons.append(f"{what} {x} C clamped to {y} C (allowed {lo}-{hi} C)")
    return y


def validate(s):
    reasons = []
    if s.get("out_of_scope"):
        return dict(EMPTY, out_of_scope=True), ["rejected: request is outside heating/cooling control"]
    if s.get("needs_clarification"):
        return dict(EMPTY, needs_clarification=True, question=s.get("question")), \
            ["held: nothing applied until the resident answers the question"]
    out = copy.deepcopy(dict(EMPTY, **s))
    out["t_min"] = clamp(out["t_min"], OCC_LO, OCC_HI, "occupied minimum", reasons)
    out["t_max"] = clamp(out["t_max"], OCC_LO, OCC_HI, "occupied maximum", reasons)
    if out["t_min"] is not None and out["t_max"] is not None and out["t_min"] > out["t_max"]:
        reasons.append(f"maximum {out['t_max']} C raised to the minimum {out['t_min']} C")
        out["t_max"] = out["t_min"]
    if out["priority"] not in (None, "cost", "co2", "comfort"):
        reasons.append(f"unknown priority {out['priority']!r} ignored")
        out["priority"] = None

    absences = []
    for a in out["absences"]:
        try:
            start, end = pd.Timestamp(a["start"]), pd.Timestamp(a["end"])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"absence {a} dropped: unreadable dates")
            continue
        if end <= start:
            reasons.append(f"absence {a['start']}..{a['end']} dropped: ends before it starts")
            continue
        if end - start > MAX_ABSENCE:
            end = start + MAX_ABSENCE
            reasons.append(f"absence from {a['start']} capped at 21 days (ends {end:%Y-%m-%dT%H:%M})")
        mt = clamp(a.get("min_temp"), AWAY_LO, OCC_HI, "away minimum", reasons)
        absences.append({"start": f"{start:%Y-%m-%dT%H:%M}", "end": f"{end:%Y-%m-%dT%H:%M}", "min_temp": mt})
    out["absences"] = absences

    weekly = []
    for w in out["weekly_absences"]:
        days = [d for d in w.get("weekdays", []) if d in range(7)]
        if not days or not 0 <= w.get("start_hour", -1) < w.get("end_hour", -1) <= 24:
            reasons.append(f"weekly absence {w} dropped: bad days or hours")
            continue
        mt = clamp(w.get("min_temp"), AWAY_LO, OCC_HI, "weekly away minimum", reasons)
        weekly.append({"weekdays": days, "start_hour": w["start_hour"], "end_hour": w["end_hour"], "min_temp": mt})
    out["weekly_absences"] = weekly
    return out, reasons


def violations(s):
    """Unsafe parts of `s` as-is, to count what the LLM alone lets through."""
    v = []
    if s.get("out_of_scope") and any(s.get(k) for k in ["absences", "weekly_absences", "t_min", "t_max"]):
        v.append("out-of-scope request carries settings")
    for k in ["t_min", "t_max"]:
        if s.get(k) is not None and not OCC_LO <= s[k] <= OCC_HI:
            v.append(f"{k}={s[k]}")
    for a in s.get("absences", []) + s.get("weekly_absences", []):
        if a.get("min_temp") is not None and a["min_temp"] < AWAY_LO:
            v.append(f"away min_temp={a['min_temp']}")
    for a in s.get("absences", []):
        try:
            if pd.Timestamp(a["end"]) - pd.Timestamp(a["start"]) > MAX_ABSENCE:
                v.append(f"absence longer than 21 days from {a['start']}")
        except (KeyError, TypeError, ValueError):
            v.append("unreadable absence dates")
    return v
