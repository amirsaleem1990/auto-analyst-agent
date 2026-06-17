#!/home/amir/github/auto-analyst-agent/.venv/bin/python3

"""
Auto-Analyst

An AI agent reads arbitrary tabular data, explores it with code when needed,
and writes its own insights. No column names or domain logic are assumed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, TypedDict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from openrouter_free_models import get_free_models

MAX_PREVIEW_ROWS = 8
MAX_CODE_OUTPUT_CHARS = 12_000
MAX_UPLOAD_ROWS = 200
MAX_TOOL_ROUNDS = 12
MAX_MODEL_TRIES = 8

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": (
                "Execute Python/pandas code to explore the dataset. "
                "Variables available: df (DataFrame), pd. "
                "Assign findings to a variable named `result` (string or printable)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to run against df.",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Brief note on what this code checks.",
                    },
                },
                "required": ["code", "purpose"],
            },
        },
    }
]

SYSTEM_PROMPT = """\
You are an auto-analyst. You receive a tabular dataset with unknown structure.

Rules:
- Do not assume what columns mean until you inspect the data.
- Use run_code to explore patterns, distributions, trends, correlations, or anomalies.
- Ground every claim in what you observed from the data or tool output.
- If the data is ambiguous, say what is unclear and what you inferred.
- Write for a business reader: clear, specific, and actionable.

When finished exploring, respond with ONLY a markdown report using these sections:
## Dataset Overview
## Key Patterns
## Notable Findings
## Risks or Data Quality Issues
## Recommended Actions
"""

all_free_models = get_free_models(tools_only=True)
all_free_models_names = [model["id"] for model in all_free_models]


class ToolCallEvent(TypedDict, total=False):
    phase: str
    tool: str
    purpose: str
    code: str
    output: str


StatusCallback = Callable[[str], None]
ToolCallCallback = Callable[[ToolCallEvent], None]


def _status(on_status: StatusCallback | None, message: str) -> None:
    if on_status:
        on_status(message)
    else:
        print(message, file=sys.stderr)

def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    readers = {
        ".csv": lambda p: pd.read_csv(p),
        ".tsv": lambda p: pd.read_csv(p, sep="\t"),
        ".json": lambda p: pd.read_json(p),
        ".parquet": lambda p: pd.read_parquet(p),
        ".xlsx": lambda p: pd.read_excel(p),
        ".xls": lambda p: pd.read_excel(p),
    }
    if suffix not in readers:
        supported = ", ".join(sorted(readers))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")

    return readers[suffix](path)


def load_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    """Load a tabular file uploaded via Streamlit."""
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    buffer = io.BytesIO(data)

    readers = {
        ".csv": lambda: pd.read_csv(buffer),
        ".tsv": lambda: pd.read_csv(buffer, sep="\t"),
        ".json": lambda: pd.read_json(buffer),
        ".parquet": lambda: pd.read_parquet(buffer),
        ".xlsx": lambda: pd.read_excel(buffer),
        ".xls": lambda: pd.read_excel(buffer),
    }
    if suffix not in readers:
        supported = ", ".join(sorted(readers))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")

    return readers[suffix]()


def profile_data(df: pd.DataFrame) -> str:
    """Schema-agnostic snapshot of the dataset for the agent."""
    lines = [
        f"Rows: {len(df)}",
        f"Columns ({len(df.columns)}): {', '.join(map(str, df.columns))}",
        "",
        "Column types:",
    ]

    for col in df.columns:
        nulls = int(df[col].isna().sum())
        unique = int(df[col].nunique(dropna=True))
        lines.append(f"- {col}: {df[col].dtype}, nulls={nulls}, unique={unique}")

    lines.append("")
    lines.append("Sample rows:")
    lines.append(df.head(MAX_PREVIEW_ROWS).to_string(index=False))

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        lines.append("")
        lines.append("Numeric summary:")
        lines.append(numeric.describe().to_string())

    non_numeric = [
        c 
        for c in df.columns 
        if c not in numeric.columns
    ]
    if non_numeric:
        for col in non_numeric:
            counts = df[col].value_counts(dropna=False).head(5)
            lines.append("")
            lines.append(f"Top values for '{col}':")
            for value, count in counts.items():
                lines.append(f"- {value!r}: {count}")

    return "\n".join(lines)


def run_exploration_code(df: pd.DataFrame, code: str) -> str:
    """Run agent-authored pandas code against the loaded frame."""
    buffer = io.StringIO()
    local_vars: dict[str, Any] = {"df": df.copy(), "pd": pd}
    try:
        compiled = compile(code, "<agent>", "exec")
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(compiled, {"__builtins__": __builtins__}, local_vars)  # noqa: S102
        if "result" in local_vars:
            output = str(local_vars["result"])
        else:
            output = buffer.getvalue().strip() or "Code ran successfully (no `result` variable set)."
    except Exception:
        output = traceback.format_exc()

    if len(output) > MAX_CODE_OUTPUT_CHARS:
        output = output[:MAX_CODE_OUTPUT_CHARS] + "\n... (truncated)"
    return output

def create_client(api_key: str) -> OpenAI:
    """OpenRouter exposes an OpenAI-compatible API."""
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
    }

    return OpenAI(**kwargs)


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or ("rate" in text and "limit" in text)


def is_tool_use_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "404" in text and "tool" in text


def parse_retry_after(exc: Exception) -> int:
    match = re.search(r"retry_after_seconds['\"]?\s*[:=]\s*(\d+)", str(exc))
    if match:
        return int(match.group(1))
    return 30


def resolve_model(model: str, on_status: StatusCallback | None = None) -> str:
    """Use the requested model if tool-capable, otherwise pick a fallback."""
    if model in all_free_models_names:
        return model
    if all_free_models_names:
        fallback = all_free_models_names[0]
        _status(
            on_status,
            f"Model {model!r} is not tool-capable on OpenRouter; using {fallback}",
        )
        return fallback
    return model


def pick_next_model(tried: set[str]) -> str | None:
    candidates = [m for m in all_free_models_names if m not in tried]
    if not candidates:
        return None
    return random.choice(candidates)


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tried_models: set[str],
    *,
    max_model_tries: int = MAX_MODEL_TRIES,
    on_status: StatusCallback | None = None,
) -> tuple[Any, str]:
    """
    Call the model; on rate limit, wait and retry/switch (up to N tries).
    """
    current_model = resolve_model(model, on_status)
    tried_models.add(current_model)

    for attempt in range(max_model_tries):
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            return response, current_model
        except Exception as exc:
            if attempt + 1 >= max_model_tries:
                raise

            if is_tool_use_error(exc):
                _status(
                    on_status,
                    f"{current_model} does not support tool use, trying another model...",
                )
                next_model = pick_next_model(tried_models)
                if not next_model:
                    raise RuntimeError(
                        "No more tool-capable free models available."
                    ) from exc
                current_model = next_model
                tried_models.add(current_model)
                continue

            if is_rate_limit_error(exc):
                wait_seconds = parse_retry_after(exc)
                _status(
                    on_status,
                    f"Rate limited on {current_model} "
                    f"(attempt {attempt + 1}/{max_model_tries}), "
                    f"waiting {wait_seconds}s...",
                )
                time.sleep(wait_seconds)

                next_model = pick_next_model(tried_models)
                if next_model:
                    _status(
                        on_status,
                        f"Switching model: {current_model} -> {next_model}",
                    )
                    current_model = next_model
                    tried_models.add(current_model)
                continue

            raise

    raise RuntimeError(f"All {max_model_tries} model attempts failed.")


def run_agent(
    df: pd.DataFrame,
    profile: str,
    model: str,
    api_key: str,
    *,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    max_model_tries: int = MAX_MODEL_TRIES,
    on_status: StatusCallback | None = None,
    on_tool_call: ToolCallCallback | None = None,
    tool_calls_log: list[ToolCallEvent] | None = None,
) -> str:
    client = create_client(api_key)
    tried_models: set[str] = set()
    current_model = model

    messages: list[dict[str, Any]] = [
        {
            "role": "system", 
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                "Analyze this dataset. Explore as needed before concluding.\n\n"
                f"Initial profile:\n{profile}"
            ),
        },
    ]

    for turn in range(max_tool_rounds):
        _status(
            on_status,
            f"Agent turn {turn + 1}/{max_tool_rounds} (model: {current_model})",
        )

        response, current_model = chat_completion(
            client,
            current_model,
            messages,
            tried_models,
            max_model_tries=max_model_tries,
            on_status=on_status,
        )
        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                args = json.loads(call.function.arguments)
                code = args.get("code", "")
                purpose = args.get("purpose", "")
                tool_name = call.function.name

                event: ToolCallEvent = {
                    "phase": "start",
                    "tool": tool_name,
                    "purpose": purpose,
                    "code": code,
                }
                if on_tool_call:
                    on_tool_call(event)

                _status(
                    on_status,
                    f"Tool call: {tool_name}\nPurpose: {purpose}\nCode:\n{code}",
                )

                tool_output = run_exploration_code(df, code)
                event = {
                    "phase": "end",
                    "tool": tool_name,
                    "purpose": purpose,
                    "code": code,
                    "output": tool_output,
                }
                if tool_calls_log is not None:
                    tool_calls_log.append(event)
                if on_tool_call:
                    on_tool_call(event)

                _status(on_status, f"Output:\n{tool_output}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_output,
                    }
                )
            continue

        if message.content:
            return message.content.strip()

    return (
        "Analysis stopped after reaching the exploration turn limit. "
        "Re-run with a smaller dataset or increase max tool rounds."
    )


def prepare_sample_df(
    df: pd.DataFrame,
    max_upload_rows: int = MAX_UPLOAD_ROWS,
    random_state: int = 44,
) -> pd.DataFrame:
    if len(df) <= max_upload_rows:
        return df.copy()
    return df.sample(max_upload_rows, random_state=random_state).reset_index(drop=True)


def analyze_dataframe(
    df: pd.DataFrame,
    model: str,
    api_key: str,
    *,
    max_upload_rows: int = MAX_UPLOAD_ROWS,
    random_state: int = 44,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    max_model_tries: int = MAX_MODEL_TRIES,
    on_status: StatusCallback | None = None,
    on_tool_call: ToolCallCallback | None = None,
) -> dict[str, Any]:
    profile = profile_data(df)
    sample_df = prepare_sample_df(df, max_upload_rows, random_state)
    tool_calls_log: list[ToolCallEvent] = []
    analysis = run_agent(
        sample_df,
        profile,
        model,
        api_key,
        max_tool_rounds=max_tool_rounds,
        max_model_tries=max_model_tries,
        on_status=on_status,
        on_tool_call=on_tool_call,
        tool_calls_log=tool_calls_log,
    )
    return {
        "profile": profile,
        "analysis": analysis,
        "sample_df": sample_df,
        "tool_calls": tool_calls_log,
        "total_rows": len(df),
        "sample_rows": len(sample_df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI auto-analyst for arbitrary tabular data.",
    )
    parser.add_argument(
        "--input",
        help="Path to CSV/TSV/JSON/Parquet/Excel file",
    )
    parser.add_argument(
        "--output",
        default="analysis.md",
        help="Path to write the markdown analysis",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        help="OpenRouter model id (e.g. meta-llama/llama-3.3-70b-instruct:free)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is required. Copy .env.example to .env and set your key."
        )

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    print(f"Loading {path}...", file=sys.stderr)
    df = load_data(path)
    profile = profile_data(df)

    print("\n\n")
    print(profile)
    big_data = (
        f"(exploring a random sample of {MAX_UPLOAD_ROWS} rows out of {len(df)} total)"
        if len(df) > MAX_UPLOAD_ROWS
        else ""
    )
    print(f"\n\nRunning auto-analyst agent {big_data} ...\n", file=sys.stderr)

    result = analyze_dataframe(df, args.model, api_key)
    analysis = result["analysis"]

    Path(args.output).write_text(analysis + "\n", encoding="utf-8")
    print(f"Analysis written to: {args.output}")


if __name__ == "__main__":
    main()
