# informed-comfort

A small working version of the Aarhus University PhD project
[Informed comfort: natural-language interaction with home heating and cooling](https://phd.tech.au.dk/for-applicants/apply-here/saeropslag/informed-comfort-natural-language-interaction-with-home-heating-and-cooling).

One simulated Danish house in Aarhus, heated by an air-to-water heat pump, runs
through the winter of 2025–26 under a predictive controller (MPC). Residents write
to it in English or Danish. An LLM turns each message into controller settings, a
deterministic validator makes them safe, and the LLM explains the plan in kr and
kg CO₂ using only numbers the controller computed.

The listing names three ways this dialogue saves energy. They "look the same on the
energy meter"; in simulation each can be switched on and off separately:

| Mechanism (listing) | Here |
|---|---|
| **Occupancy:** heat only when people are home | Exp 1: absences from messages feed the MPC's comfort band |
| **Personal comfort:** setpoints that match each household | Exp 1–2: band changes from messages; band-width sweep. *Stated* preferences only |
| **Energy sufficiency:** residents knowingly accept less | Exp 3: a daily "1 °C lower today" offer with its what-if saving, accepted with assumed probability p |

> **LLM status.** LLM results come from the hosted DeepSeek API (`deepseek-v4-pro`).
> Local Ollama is supported but has not been run. LLM accuracy is **provisional**: the
> 99 gold labels in `requests.jsonl` have not been reviewed (check time rounding, the
> "target X → band X±1" rule, and ambiguous vs out of scope). The STUB only tests the pipeline.

## Real vs simulated

- **Real:** DK1 day-ahead price, realised CO₂ intensity and Energinet's CO₂ forecast
  (Energi Data Service: `DayAheadPrices`, `CO2Emis`, `CO2EmisProg`); hourly Aarhus
  temperature and solar radiation (Open-Meteo archive). All resampled to hourly means.
- **Simulated / assumed:** the house, the heat pump, three households, their schedules
  and all 99 messages.
- `data.py` also handles the retired `Elspotprices` dataset for windows before
  2025-10-01; this window only needed `DayAheadPrices`.

## Assumed parameters

| Parameter | Value | Why |
|---|---|---|
| Heat loss 1/R | 150 W/K (R = 6.67 K/kW) | ~150 m² house, partly renovated |
| Heat capacity C | 10 kWh/K | medium-heavy house; RC = 67 h |
| COP | 2.0 at −5 °C, 3.5 at 7 °C, linear, clamped to [1.5, 4.0] | typical air-to-water datasheet at ~35 °C flow |
| P_max | 4 kW electrical | holds 22 °C at the coldest hour (−10 °C) |
| Solar aperture | 2 m² | low winter sun, small windows |
| Comfort band | 20–22 °C | 21 ± 1 °C |
| Away floor | 16 °C | when a resident gives none |
| λ (kr per kg CO₂) | 0.75; 0 for "cost", 5 for "co2"; "comfort" adds 0.5 °C to the lower edge | 0.75 = 2030 Danish CO₂ tax, non-ETS (750 DKK/t) |
| Slack penalty M | 100 kr per °C·h | the band breaks only when it cannot be kept |
| Safety limits | occupied 16–24 °C; away ≥ 12 °C; absence ≤ 21 days | health, frost and mould; long absences need a human |
| Electricity price | spot only | no tariffs, taxes or VAT |

## Method

**House.** One step is one hour:

    T_{t+1} = T_t − (T_t − T_out,t)·Δt/(RC) + COP(T_out,t)·P_t·Δt/C + A_sol·I_t·Δt/C

**MPC.** Every hour, a linear program plans electrical power P_t for the next 24 h:

    min  Σ P_t·Δt·(price_t + λ·CO₂forecast_t) + M·Σ s_t
    s.t. T_min,t − s_t ≤ T_t ≤ T_max,t + s_t,   0 ≤ P_t ≤ P_max,   s_t ≥ 0

COP depends only on outdoor temperature, so T is linear in P (T = G·P + f,
`mpc.predict`). Actual weather is used as a **perfect forecast**. Only the first hour
is applied, then the LP is solved again.

**Settings → bands.** When nobody is home, T_min drops to the away floor. The LP sees
the return time and preheats in cheap hours. A message is known only from the hour it
was sent.

**Baselines.** *fixed_21C*: a perfect thermostat at 21 °C. *rule_cheap25*: band top
in the cheapest 25 % of each day's hours, else band bottom; ignores absences.

**What-if.** Re-solves the same 24 h with one setting changed and returns the
difference in kr, kg CO₂ and comfort. The alternative MUST end at least as warm, so a
one-off coast-down is not counted as a saving.

**LLM.** Ollama with a JSON schema, or DeepSeek in JSON mode; temperature 0. It writes
down what was asked, even if extreme; `safety.py` decides what is allowed.
`explain.check()` requires every number with kr/kg/kWh/°C in an explanation to match a
computed value (±0.06 or ±2 %).

**Scoring.** Cost and CO₂ use the **actual** series (the MPC plans with the CO₂
forecast). Comfort is degree-hours outside the band while someone is truly home.

## Data used

From the Energi Data Service and Open-Meteo data above (summary in results/data_info.json):

- 2025-12-01 to 2026-02-28 (Danish time), 2160 hours, no hour missing.
- Price mean 0.71 kr/kWh (0.02–2.07). CO₂ mean 113.6 g/kWh, forecast MAE 4.5 g/kWh.
  Price–CO₂ correlation 0.375.
- Outdoor temperature mean 1.5 °C (−10.0 to 11.5 °C).

## Results

One winter (90 days), three households, ground-truth settings unless stated.

**Experiment 1: occupancy** (`exp1_occupancy.csv`; violation in °C·h, occupied hours)

| Household | Controller | kWh | kr | kg CO₂ | violation |
|---|---|---|---|---|---|
| commuter couple | fixed_21C | 2371 | 1735 | 289 | 0.0 |
| | rule_cheap25 | 2385 | 1672 | 289 | 1.3 |
| | mpc_no_occupancy | 2367 | 1573 | 277 | 0.7 |
| | mpc_ground_truth | 2283 | 1507 | 262 | 0.7 |
| home worker | fixed_21C | 2371 | 1735 | 289 | 0.0 |
| | rule_cheap25 | 2452 | 1663 | 296 | 0.0 |
| | mpc_no_occupancy | 2425 | 1542 | 288 | 0.0 |
| | mpc_ground_truth | 2402 | 1526 | 285 | 0.0 |
| family with kids | fixed_21C | 2371 | 1735 | 289 | 0.0 |
| | rule_cheap25 | 2406 | 1703 | 292 | 0.0 |
| | mpc_no_occupancy | 2423 | 1659 | 291 | 0.0 |
| | mpc_ground_truth | 2300 | 1541 | 266 | 0.0 |

- **MPC beats both baselines on cost in every household:** 11–13 % cheaper than fixed
  21 °C with ground-truth absences.
- **Occupancy adds 4.2 / 1.0 / 7.1 %** (commuter / home worker / family). It is modest
  because the house is heavy (RC = 67 h) and must be back in the band on return.
- **Price-shifting alone can use more energy.** For the home worker and the family,
  mpc_no_occupancy and rule_cheap25 use 1.5–3.4 % *more* kWh than fixed 21 °C, and CO₂
  does not fall: λ is small next to the price.
- All commuter violations come from one "at least 21 °C" message on 5 January that
  arrived when the house was below 21 °C. No controller could avoid them.

**Experiment 2: comfort band 21 ± w °C** (`exp2_band_sweep.csv`)

| w (°C) | commuter kr | home worker kr | family kr |
|---|---|---|---|
| 0.5 | 1501 | 1538 | 1505 |
| 1.0 | 1433 | 1445 | 1424 |
| 1.5 | 1381 | 1384 | 1367 |
| 2.0 | 1338 | 1334 | 1322 |

±0.5 → ±2 °C cuts cost by 11–13 % and CO₂ by 6–9 %, mostly via the lower edge.

**Experiment 3: "1 °C lower today" offers** (`exp3_sufficiency.csv`; saving vs p = 0)

| p | commuter kr (kg) | home worker kr (kg) | family kr (kg) |
|---|---|---|---|
| 0.25 | 18.2 (1.8) | 17.8 (3.1) | 28.6 (3.3) |
| 0.5 | 43.1 (7.5) | 42.6 (6.3) | 54.8 (7.9) |
| 0.75 | 71.2 (10.7) | 67.0 (9.2) | 93.6 (10.9) |
| 1.0 | 105.6 (15.9) | 103.0 (13.5) | 127.9 (14.7) |

- At p = 1 the saving is 103–128 kr per winter (7–8 % of MPC cost).
- **The quote is honest to within ~20 %.** At p = 1 the summed quotes are 19–22 % low,
  because each quote assumes the house warms back up the next day.

**Experiment 4: LLM** (DeepSeek V4 Pro, provisional; `llm_eval_deepseek_deepseek-v4-pro.csv`)

| Metric | Value |
|---|---|
| All fields correct | 92/99 |
| t_min, t_max, out_of_scope | 1.000 each |
| absences / weekly / priority / needs_clarification | 0.970 / 0.980 / 0.970 / 0.970 |
| Clarification recall (ambiguous messages) | 10/13 |
| Unsafe settings: LLM alone → after validator | 13 → **0** |
| Explanations fully faithful | 27/27 |
| Parse latency mean / p95; parse errors | 3.2 s / 5.9 s; 0 |

- Weakest on ambiguous messages: it sometimes fills in times instead of asking
  ("We're out tomorrow").
- **Cost of parsing errors** (`exp1_occupancy_with_llm_*.csv`): none for the home worker,
  0.2 kr for the family. The commuter gets 41 of the 66 kr occupancy is worth: a weekly
  Mon–Fri absence was parsed as five one-off days.

Plots: `results/exp1_occupancy.png`, `exp2_band_sweep.png`, `exp3_sufficiency.png`,
`week_trace.png`.

## Calibration

`calibrate.py` fits R, C and A_sol to one real building and tests on held-out data.

**Data.** Sartori, ..., Knudsen M.D., et al. (2023), *Sub-hourly measurement datasets
from 6 real buildings*, Data in Brief 48, 109149; data doi:10.17632/xztfbtsgys.3 (CC BY 4.0).
Dataset 2, the ZEB Living Lab (NTNU, Trondheim): an unoccupied, highly insulated test
house with one metered radiator. **Not Danish, no heat pump**, so COP and P_max stay assumed.

**Method.** Least squares on `T[k+1]−T[k] = a·(Tout−T) + b·Q + c·I` (hourly). Split fixed
before fitting: train 10–28 Feb 2019 (PRBS phase, 377 h), validate open-loop
1–13 Mar 2019 (293 h).

| | R (K/kW) | UA (W/K) | C (kWh/K) | τ (h) | A_sol (m²) |
|---|---|---|---|---|---|
| assumed | 6.67 | 150 | 10.0 | 67 | 2.0 |
| fitted | 11.64 | 86 | 4.94 | 57 | 2.0 |

| Held-out, open loop | RMSE (°C) | max error (°C) |
|---|---|---|
| fitted | **2.55** | 4.84 |
| assumed | 9.87 | 13.20 |

The solar term is not identified (SE ±82 %). The fit tracks the first ~4 held-out days,
then runs ~3 °C too cold in a colder, sunnier spell than training. Nothing was tuned
after seeing this. **An order-of-magnitude calibration, not a validated model.** Next
steps: 2R2C, wider training weather, room sensors instead of extract air.

**Effect on the experiments** (`*_calibrated.*`). The fitted house uses 43 % less energy
(fixed 21 °C: 1341 vs 2371 kWh). Relative results barely move:

| | assumed | calibrated |
|---|---|---|
| Exp 1: MPC vs fixed 21 °C, cost | −11 to −13 % | −11 to −13 % |
| Exp 1: added by occupancy (commuter / home worker / family) | 4.2 / 1.0 / 7.1 % | 5.0 / 1.3 / 7.7 % |
| Exp 2: cost cut ±0.5 → ±2 °C | 11–13 % | 11–14 % |
| Exp 3: saving at p = 1 | 103–128 kr | 61–72 kr |
| Exp 3: quote ÷ simulated at p = 1 | 78–81 % | 83–85 % |
| Exp 4: commuter occupancy value kept with LLM parsing | 41 of 66 kr | 25 of 45 kr |

## Limitations

- **No people.** Households and messages are invented. Acceptance, and how explanations
  change judged comfort, need the field study.
- **Assumed house.** Parameters are assumptions; the one fit is to a Norwegian test
  house with a poor held-out fit. A lighter house would gain more from occupancy.
- **Perfect forecasts.** Weather and all 24 h of prices are known; only CO₂ is a real
  forecast. Model and simulator are the same equations.
- **Assumed p.** Exp 3 shows savings for a given p, independent day to day.
- **Hosted LLM.** Fine for invented messages. Real messages reveal when a home is empty
  and would need a local model or EU hosting (GDPR).
- **Small, self-labelled test set.** One LLM, temperature 0, 99 messages labelled by
  their author.
- **Spot price only.** Grid tariffs, taxes and VAT change the price shape.
- **Average CO₂, not marginal.** Realised and forecast series use different bases.
- **One winter, one seed, one region (DK1), one zone, heating only.** Comfort is stated,
  not learned.

## How to run

### Mac / Linux

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python data.py                              # fetch once; then offline
.venv/bin/python -m pytest -q                         # spec checks
.venv/bin/python run.py --skip-llm                    # experiments 1-3
.venv/bin/python run.py --stub                        # also exp 4 with the STUB
LLM_BACKEND=deepseek .venv/bin/python run.py          # exp 4 with DeepSeek (see below)
.venv/bin/python -I calibrate.py                      # fit to the ZEB house
HOUSE=calibrated .venv/bin/python run.py --skip-llm   # experiments with the fitted house
```

| File | Role |
|---|---|
| `data.py` | fetch, cache, align → `results/data_info.json` |
| `house.py` | 1R1C house and COP |
| `mpc.py` | settings → bands, the LP, `what_if()`, baselines, simulation, scoring |
| `llm.py` | Ollama or DeepSeek client, cache, STUB |
| `safety.py` | validator, one reason per change |
| `explain.py` | explanations and faithfulness check |
| `calibrate.py` | fit R, C, A_sol to a real building |
| `requests.jsonl` | 99 messages with gold labels |
| `run.py` | experiments 1–4 → `results/` |
| `test_checks.py` | spec checks |

### DeepSeek API

Put one line `DEEPSEEK_API_KEY=...` in `.env` in the project folder. It MUST NOT be
shared; the code never prints or stores it. `DEEPSEEK_MODEL` picks the model (default
`deepseek-v4-pro`). Thinking is disabled, so it acts as a plain chat model. JSON mode
guarantees valid JSON but not the schema, so the schema also goes in the prompt.
Every call is cached in `results/llm_cache.json`.

### Windows + Ollama (local model)

On the Windows PC (RTX 4070 Ti 12 GB), in PowerShell from the project folder:

1. Install Ollama from <https://ollama.com/download>, then `ollama pull qwen2.5:14b`.
2. Run:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m pytest -q
   $env:OLLAMA_MODEL = "qwen2.5:14b"
   python run.py
   ```
   `data/` is already filled. If `Activate.ps1` is blocked, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
3. Results go to `results/*_<model>.csv`. Calls are cached with their latency; delete
   the cache to measure latency again.

| Candidate model | Size (Q4_K_M) | Why |
|---|---|---|
| `qwen2.5:14b` | 9.0 GB | strong at structured JSON; multilingual |
| `gemma3:12b` | 8.1 GB | multilingual; good instruction following |
| `qwen3:14b` | 9.3 GB | newer "thinking" model; structured output not tested here |
