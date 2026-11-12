"""Turn a resident's message into a Settings dict with an LLM.

Backend (env LLM_BACKEND): "ollama" (default, a local model) or "deepseek" (the
hosted DeepSeek API, key read from DEEPSEEK_API_KEY or the project's .env file).

Real mode: POST {OLLAMA_URL}/api/chat with `format` = the JSON schema below, so
Ollama constrains the model's output to valid Settings JSON. Every call is
cached in results/llm_cache.json, keyed by model + prompt, together with its
measured latency, so a rerun gives the same numbers without the GPU.

STUB mode (`--stub` or LLM_STUB=1): no model is called. parse() returns the
gold label with a deterministic corruption chosen by the request id:
    id % 10 == 3  -> needs_clarification flipped (a missed / spurious question)
    id % 10 == 7  -> every absence ends 24 h late; weekly absences end 1 h late
    id % 10 == 9  -> priority dropped; t_min, if set, 1 C lower
Stub outputs test the pipeline, NOT a model; everything they produce is labelled STUB.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from safety import EMPTY

CACHE = Path(__file__).parent / "results" / "llm_cache.json"
ENV_FILE = Path(__file__).parent / ".env"
STUB_MODEL = "STUB"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_DEFAULT = "deepseek-v4-pro"    # api-docs.deepseek.com, checked 2026-11-09 (deepseek-chat retired 2026-07-24)
RETRIES = 3                             # extra tries on 429 / 5xx, waiting 2, 4, 8 s

ABSENCE = {"type": "object", "required": ["start", "end", "min_temp"], "properties": {
    "start": {"type": "string"}, "end": {"type": "string"}, "min_temp": {"type": ["number", "null"]}}}
WEEKLY = {"type": "object", "required": ["weekdays", "start_hour", "end_hour", "min_temp"], "properties": {
    "weekdays": {"type": "array", "items": {"type": "integer"}},
    "start_hour": {"type": "integer"}, "end_hour": {"type": "integer"},
    "min_temp": {"type": ["number", "null"]}}}
SCHEMA = {"type": "object", "required": list(EMPTY), "properties": {
    "absences": {"type": "array", "items": ABSENCE},
    "weekly_absences": {"type": "array", "items": WEEKLY},
    "t_min": {"type": ["number", "null"]}, "t_max": {"type": ["number", "null"]},
    "priority": {"enum": ["cost", "co2", "comfort", None]},
    "needs_clarification": {"type": "boolean"}, "question": {"type": ["string", "null"]},
    "out_of_scope": {"type": "boolean"}}}

SYSTEM = """You turn a resident's message to their home heating system into JSON settings.
Messages may be in English or Danish. Times are Danish local time.
Rules:
- absences: one-off periods nobody is home. start/end as "YYYY-MM-DDTHH:MM". Round the start UP and
  the end DOWN to a full hour. "now" means the current time rounded up to the full hour.
- weekly_absences: repeating periods. weekdays 0=Monday ... 6=Sunday. start_hour inclusive, end_hour
  exclusive, rounded like absences.
- min_temp: only if they name a temperature for while they are away; "heating off" means 0. Else null.
- t_min / t_max: the comfort band while someone is home. One target temperature X means t_min = X-1 and
  t_max = X+1. "at least X" sets only t_min; "at most X" sets only t_max.
- priority: "cost", "co2" or "comfort" if they say what matters most, else null.
- Write down exactly what they asked, even if it seems extreme. A separate safety check enforces limits.
- If something needed is missing (when, how long, how much), set needs_clarification true, ask ONE short
  question in "question", and leave all other settings empty.
- out_of_scope is true if the message is not about heating or cooling the rooms of the home (lights,
  appliances, hot water, car charging, general questions). Then leave all other settings empty.
Unused fields: empty list, null or false."""


def cache_key(model, messages, schema):
    return hashlib.sha256(json.dumps([model, messages, schema], sort_keys=True).encode()).hexdigest()


def cache_load():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def cache_put(key, value):
    c = cache_load()
    c[key] = value
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(c, indent=1, ensure_ascii=False))


def backend():
    b = os.environ.get("LLM_BACKEND", "ollama")
    if b not in ("ollama", "deepseek"):
        raise RuntimeError(f"LLM_BACKEND must be ollama or deepseek, not {b!r}")
    return b


def model_name():
    """Bare name for Ollama, so existing caches stay valid."""
    if backend() == "deepseek":
        return "deepseek:" + os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT)
    return os.environ.get("OLLAMA_MODEL", "")


def chat(messages, schema=None):
    import requests

    model = model_name()
    if not model:
        raise RuntimeError("set OLLAMA_MODEL (or use --stub / LLM_STUB=1)")
    key = cache_key(model, messages, schema)
    hit = cache_load().get(key)
    if hit:
        return hit["content"], hit["latency_s"]
    if backend() == "deepseek":
        content, latency, usage = deepseek_chat(messages, schema)
        cache_put(key, {"model": model, "content": content, "latency_s": latency, "usage": usage})
        return content, latency
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
    body = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}}
    if schema:
        body["format"] = schema
    t0 = time.perf_counter()
    r = requests.post(url, json=body, timeout=600)
    r.raise_for_status()
    latency = time.perf_counter() - t0
    content = r.json()["message"]["content"]
    cache_put(key, {"model": model, "content": content, "latency_s": latency})
    return content, latency


def deepseek_key():
    """The key MUST NOT be printed or stored."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "DEEPSEEK_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing: put DEEPSEEK_API_KEY=... in .env")
    return key


def deepseek_request(messages, schema, model, key):
    """JSON mode guarantees valid JSON, not our schema, so the schema goes in the
    system message. For JSON mode the prompt MUST contain the word "json"."""
    if schema:
        assert messages[0]["role"] == "system"
        extra = "\n\nReply with a single json object matching this schema:\n" + json.dumps(schema, separators=(",", ":"))
        messages = [dict(messages[0], content=messages[0]["content"] + extra)] + messages[1:]
    body = {"model": model, "messages": messages, "temperature": 0, "stream": False,
            "thinking": {"type": "disabled"}}      # thinking is on by default; off = plain chat model
    if schema:
        body["response_format"] = {"type": "json_object"}
    return {"Authorization": f"Bearer {key}"}, body


def deepseek_chat(messages, schema):
    import requests

    headers, body = deepseek_request(messages, schema, os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT),
                                     deepseek_key())
    for attempt in range(RETRIES + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(DEEPSEEK_URL, headers=headers, json=body, timeout=120)
        except (requests.Timeout, requests.ConnectionError):   # a stalled call: same as a 5xx
            if attempt == RETRIES:
                raise
            time.sleep(2 * 2 ** attempt)
            continue
        latency = time.perf_counter() - t0
        if (r.status_code == 429 or r.status_code >= 500) and attempt < RETRIES:
            time.sleep(2 * 2 ** attempt)
            continue
        break
    if r.status_code == 401:
        raise RuntimeError("DeepSeek returned 401: the API key in .env was rejected")
    if r.status_code != 200:
        raise RuntimeError(f"DeepSeek returned {r.status_code}: {r.text[:500]}")
    j = r.json()
    return j["choices"][0]["message"]["content"], latency, j.get("usage")


def prompt(text, sent):
    now = pd.Timestamp(sent)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Current time: {now:%A %Y-%m-%d %H:%M}.\nMessage: {text}"}]


def parse(req, stub=False):
    if stub:
        return stub_parse(req), 0.0, STUB_MODEL
    content, latency = chat(prompt(req["text"], req["sent"]), SCHEMA)
    try:
        s = dict(EMPTY, **json.loads(content))
    except (json.JSONDecodeError, TypeError):
        s = dict(EMPTY, parse_error=True)   # counts as wrong on every field
    return s, latency, model_name()


def stub_parse(req):
    s = json.loads(json.dumps(gold(req)))      # deep copy
    k = req["id"] % 10
    if k == 3:
        s["needs_clarification"] = not s["needs_clarification"]
        s["question"] = "Could you tell me more?" if s["needs_clarification"] else None
    elif k == 7:
        for a in s["absences"]:
            a["end"] = f"{pd.Timestamp(a['end']) + pd.Timedelta(hours=24):%Y-%m-%dT%H:%M}"
        for w in s["weekly_absences"]:
            w["end_hour"] = min(w["end_hour"] + 1, 24)
    elif k == 9:
        s["priority"] = None
        if s["t_min"] is not None:
            s["t_min"] -= 1
    return s


def gold(req):
    """requests.jsonl stores only non-default fields; fill in the rest."""
    g = dict(EMPTY, **req["gold"])
    g["absences"] = [dict({"min_temp": None}, **a) for a in g["absences"]]
    g["weekly_absences"] = [dict({"min_temp": None}, **w) for w in g["weekly_absences"]]
    return g


def load_requests(path=Path(__file__).parent / "requests.jsonl"):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
