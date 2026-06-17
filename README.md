\# Auto-Analyst

An AI-powered data analyst that reads **any tabular dataset**, explores it with code, and writes its own insights — without assuming column names, schemas, or business rules.

Built with **Python 3.12.3**.

## Demo output

You can view a sample run in [`Output.html`](Output.html) in this repository. It shows the Streamlit UI, dataset profile, live tool calls, and the generated analysis on the [US Counties COVID-19 dataset](https://www.kaggle.com/datasets/fireballbyedimyrnmom/us-counties-covid-19-dataset).

## What it does

1. Loads a tabular file (CSV, TSV, JSON, Parquet, or Excel).
2. Builds a schema-agnostic profile (shape, types, samples, descriptive stats) from the full dataset.
3. Hands control to an LLM agent (via [OpenRouter](https://openrouter.ai/)) that can run pandas code through a `run_code` tool.
4. Produces a markdown report with overview, patterns, findings, risks, and recommended actions.

The agent does **not** use hardcoded analysis rules. It inspects the data first, explores with code, then writes conclusions grounded in what it observed.

For large files, the agent runs exploratory code on a **random sample** (default: 200 rows), while the initial profile reflects the full dataset.

## Features

- Schema-agnostic — works on unknown datasets
- Agent-driven exploration via `run_code` (pandas)
- CLI and Streamlit web UI
- Live tool-call visibility in the browser (purpose, code, output)
- OpenRouter integration with automatic retry on rate limits
- Fallback across free tool-capable models

## Project structure

| File | Description |
|---|---|
| `auto_analyst.py` | Core agent logic and CLI |
| `streamlit_app.py` | Browser UI |
| `openrouter_free_models.py` | Lists free OpenRouter models |
| `Output.html` | Sample run output (open in a browser) |
| `.env.example` | Environment variable template |
| `requirements.txt` | Python dependencies |

## Requirements

- Python **3.12.3**
- An [OpenRouter API key](https://openrouter.ai/keys)

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
```

## Usage

### Streamlit (recommended)

```bash
streamlit run streamlit_app.py
```

Upload a file in the sidebar, choose options (model, sample size, agent turns), and click **Run analysis**.

### CLI

```bash
python auto_analyst.py --input us-counties.csv --output analysis.md
```

Optional flags:

```bash
python auto_analyst.py --input data.csv --output report.md --model meta-llama/llama-3.3-70b-instruct:free
```

## Configuration

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |
| `OPENROUTER_MODEL` | No | Model id (default: `meta-llama/llama-3.3-70b-instruct:free`) |
| `OPENROUTER_BASE_URL` | No | API base URL (default: `https://openrouter.ai/api/v1`) |

## How the agent works

```
Upload data → Profile (full dataset stats)
                    ↓
              LLM agent reads profile
                    ↓
              run_code (pandas on sample) → results
                    ↓
              More exploration or final markdown report
```

The agent can call `run_code` multiple times. Each call runs Python/pandas against a sampled `df` and returns the result to the model before the next step.

## Dataset used in the sample output

- **US Counties COVID-19 Dataset** — [Kaggle](https://www.kaggle.com/datasets/fireballbyedimyrnmom/us-counties-covid-19-dataset)

## 100-word summary

I built an Auto-Analyst in Python that interprets unknown datasets without assuming schema or business context. The script loads tabular files generically, creates an objective data profile (columns, types, samples, and descriptive stats), and hands control to an LLM agent. The agent can execute pandas exploration code through a `run_code` tool to uncover trends, correlations, outliers, and data-quality issues before writing conclusions. Final output is a markdown report with overview, patterns, findings, risks, and recommended actions grounded in observed evidence. A Streamlit UI provides file upload, configurable options, and live visibility into each tool call.

## License

MIT
