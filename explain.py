"""The LLM explains a 24 h plan and one alternative, using ONLY numbers the
controller computed; a faithfulness check then compares every number in the
text that carries a unit (kr, kg, kWh, C) with those computed values.

STUB mode writes the explanation from a fixed template instead of a model, and
plants a wrong number (kr saving + 1.0) when stub_id % 5 == 2, so the check has
something to catch. Stub explanations are labelled STUB by the caller.
"""
import json
import re

from llm import chat

TOL_ABS, TOL_REL = 0.06, 0.02    # passes "23.5" for 23.47; rejects anything a rounding can't explain
UNIT = {"kr": "kr", "kg": "kg", "kwh": "kWh", "°c": "C", "c": "C", "grader": "C"}
# A number not glued to a letter/digit (so the 2 in "CO2" is skipped), then a unit.
NUM = re.compile(r"(?<![\w.,])(-?\d+(?:[.,]\d+)?)\s*(kr|kg|kwh|°\s?c|grader)\b", re.IGNORECASE)

SYSTEM = """You explain a home heating plan to the resident in 2-3 plain sentences.
Use ONLY the numbers given in the JSON, rounded to one decimal, with their units (kr, kg CO2, kWh, °C).
Do not compute new numbers. Mention the plan and the alternative."""


def facts(plan, diff, change_c=1.0):
    """plan: what_if()[0]['now']; diff: what_if()[1]. Values are (number, unit)."""
    return {"plan_kwh": (plan["kwh"], "kWh"), "plan_kr": (plan["kr"], "kr"), "plan_kg_co2": (plan["kg"], "kg"),
            "plan_lowest_temp": (plan["min_T"], "C"), "alternative_degrees_lower": (change_c, "C"),
            "alternative_saves_kr": (-diff["kr"], "kr"), "alternative_saves_kg_co2": (-diff["kg"], "kg"),
            "alternative_degree_hours_below_band": (diff["cold_dh"], "C")}


def explain(f, stub_id=None):
    v = {k: x for k, (x, _) in f.items()}
    if stub_id is not None:
        save_kr = v["alternative_saves_kr"] + (1.0 if stub_id % 5 == 2 else 0.0)   # planted error
        text = (f"Over the next 24 hours I plan to use {v['plan_kwh']:.1f} kWh, costing {v['plan_kr']:.1f} kr "
                f"and emitting {v['plan_kg_co2']:.1f} kg CO2, never below {v['plan_lowest_temp']:.1f} °C. "
                f"If you accept {v['alternative_degrees_lower']:.1f} °C lower, you would save {save_kr:.1f} kr "
                f"and {v['alternative_saves_kg_co2']:.1f} kg CO2.")
        return text, 0.0
    numbers = {k: round(x, 2) for k, x in v.items()}
    return chat([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": json.dumps(numbers)}])


def check(text, f):
    """Every (number, unit) MUST match a computed value. Signs are ignored ("saves 3 kr" vs -3)."""
    bad = []
    for num, unit in NUM.findall(text):
        x = abs(float(num.replace(",", ".")))           # Danish decimal comma
        u = UNIT[unit.lower().replace(" ", "")]
        if not any(abs(x - abs(val)) <= max(TOL_ABS, TOL_REL * abs(val)) for val, uu in f.values() if uu == u):
            bad.append(f"{num} {unit}")
    return not bad, bad
