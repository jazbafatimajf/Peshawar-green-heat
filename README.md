# Peshawar Green-Heat Mapper

Local validation + decision layer for urban greening under heat stress in Peshawar, Pakistan.

**What is original here**: not another generic tree-cover map, but (1) local validation against hand labels and NDVI baseline, (2) spatial hold-out heat correlation, (3) constrained optimiser + agent explanation + human approval, (4) retrieval restricted to a vetted species list.

## Scope (matches the brief)

| Component | Status | Notes |
|-----------|--------|-------|
| Segmentation (U-Net style + NDVI baseline) | Implemented | TinyUNet + NDVI threshold; train on ~100 hand labels |
| Heat link (green ↔ Landsat LST) | Implemented | Linear model + spatial hold-out; correlation only |
| Planner under budget / water | Implemented | Greedy optimiser + uncertainty range |
| Agent + approval | Implemented | Tool-using LLM agent (Anthropic), guardrails, persisted `/approve` log |
| Species / maintenance RAG | Implemented (starter KB) | TF-IDF retrieval over `knowledge/*.md`; citations must come from retrieval. KB text is starter notes, replace with real KP Forest Dept / paper excerpts |
| Change over time | Implemented | NDVI difference lost/gained masks |
| FastAPI + Docker | Implemented | Ready |
| Map UI | Minimal | Use Folium or any front-end against the API |
| LangSmith | Implemented | Set `LANGSMITH_TRACING=true` + key; agent run and LLM calls are traced |

**Cut (as specified)**: street-level planting, individual tree-species recognition from imagery, real-time streams, mobile app.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements-api.txt        # light + fast; use requirements.txt only if you need torch (TinyUNet training)
cp .env.example .env                                    # Windows: copy .env.example .env
# open .env and paste your key after GROQ_API_KEY=
export PYTHONPATH=.                                     # Windows PowerShell: $env:PYTHONPATH="."
uvicorn api.main:app --reload --port 8000
```

**Do the setup (venv + pip install) only once.** Next time, just activate the venv and start uvicorn. Reinstall only if `requirements*.txt` changes. Editing code needs no reinstall, and `--reload` restarts the server for you.

```bash
source .venv/bin/activate && export PYTHONPATH=. && uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000** for the web app (dark UI: choose area → plan → map → approve). The raw API test page is at http://localhost:8000/docs. In the web app, click **Create plan**; or in /docs run `POST /plan`, and check `agent.mode` in the response: `llm` means Groq answered, `template` means it fell back (see `agent.fallback_reason`).

## Quick start (demo)

```bash
# From project root
pip install -r requirements-api.txt        # light + fast; use requirements.txt only if you need torch (TinyUNet training)
export PYTHONPATH=.
uvicorn api.main:app --reload --port 8000
```

Then:

- `GET /demo/overview` – synthetic green + LST calibrated to published Peshawar ranges
- `POST /plan` – `{"budget": 500}` → selected blocks, cooling estimate ± uncertainty, agent text
- `POST /approve` – human-in-the-loop (logged), `GET /approvals`
- `GET /species` – vetted list only
- `GET /validation/summary` – the honesty checks

Docker:

```bash
docker build -t peshawar-green-heat .
docker run --env-file .env -p 8000:8000 peshawar-green-heat
```

## Agent, guardrails and tracing

`src/agent.py` (Groq `openai/gpt-oss-120b` by default via its OpenAI-compatible API, or Anthropic) runs a tool-using loop (`get_plan_summary`, `list_species`, `search_knowledge`) and the model must end with `submit_recommendation`. Guardrails are enforced in code:

- every species must be in the vetted list
- every citation must be an id retrieved in that same run
- the caveat must state the estimate is correlational/uncertain
- all numbers in the final text come from the planner, never from the model

If any check fails, or there is no API key, the API returns the deterministic template and says so in `agent.mode` / `agent.fallback_reason`. Copy `.env.example` to `.env` for keys. Approvals are appended to `artifacts/approvals.jsonl` (`GET /approvals`). Tests: `PYTHONPATH=. pytest -q`.

## Data pipeline (real imagery)

```bash
python -m src.data_access          # downloads a low-cloud Sentinel-2 window over Peshawar via Planetary Computer
# Then compute NDVI, run baseline / model, link to Landsat LST (same STAC client)
```

Planetary Computer STAC is free and requires no key for search; SAS signing is used for COG reads.

## Hard checks (must report honestly)

1. Does TinyUNet beat the NDVI-threshold baseline on held-out hand-labelled tiles? If not, say so and lean on the planner + change maps.
2. Does green cover explain LST on **neighbourhood** hold-outs (not random pixels)? Report R² and caveat that this is correlation, not causal proof of planting cooling.
3. Cooling estimates carry an uncertainty range derived from model fit quality; public thermal data is coarse (~100 m).



## Project layout

```
peshawar-green-heat/
├── api/main.py          # FastAPI
├── src/
│   ├── config.py
│   ├── data_access.py   # Planetary Computer STAC → Sentinel-2 / Landsat
│   ├── segmentation.py  # NDVI baseline + TinyUNet
│   ├── heat_link.py     # correlation + hold-out
│   ├── planner.py       # optimiser + agent text + approval
│   └── change.py        # multi-year green loss/gain
├── data/                # downloaded windows & labels
├── models/              # TinyUNet weights
├── notebooks/           # evaluation & failure cases
├── docs/
├── Dockerfile
├── requirements.txt
└── README

## Citations / prior work

Urban heat and green cover in Peshawar are already studied (Landsat LST / UTFVI time series). This project’s contribution is the local validation protocol, the constrained decision layer, and the honest uncertainty + human-approval loop for planners.
