#!/usr/bin/env python3
"""Streamlit UI for the Auto-Analyst."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from auto_analyst import (
    MAX_MODEL_TRIES,
    MAX_TOOL_ROUNDS,
    MAX_UPLOAD_ROWS,
    all_free_models_names,
    analyze_dataframe,
    load_uploaded_file,
)

load_dotenv()

st.set_page_config(page_title="Auto-Analyst", page_icon="📊", layout="wide")

st.title("Auto-Analyst")
st.caption("Upload tabular data and let the AI agent explore it and write insights.")

default_model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
env_api_key = os.getenv("OPENROUTER_API_KEY", "")

with st.sidebar:
    st.header("Options")

    uploaded_file = st.file_uploader(
        "Data file",
        type=["csv", "tsv", "json", "parquet", "xlsx", "xls"],
    )

    use_custom_model = st.checkbox("Use custom model id", value=False)
    if use_custom_model:
        model = st.text_input("Model id", value=default_model)
    else:
        model_options = list(dict.fromkeys([default_model, *all_free_models_names]))
        model = st.selectbox("Model", model_options)

    api_key = st.text_input(
        "OpenRouter API key",
        value=env_api_key,
        type="password",
        help="Uses OPENROUTER_API_KEY from .env if set.",
    )

    max_upload_rows = st.number_input(
        "Sample rows for agent",
        min_value=10,
        max_value=10_000,
        value=MAX_UPLOAD_ROWS,
        step=10,
    )
    random_state = st.number_input(
        "Random seed",
        min_value=0,
        value=44,
        step=1,
    )
    max_tool_rounds = st.slider(
        "Max agent turns",
        min_value=1,
        max_value=30,
        value=MAX_TOOL_ROUNDS,
    )
    max_model_tries = st.slider(
        "Max model retries (rate limits)",
        min_value=1,
        max_value=20,
        value=MAX_MODEL_TRIES,
    )

    run = st.button("Run analysis", type="primary", use_container_width=True)

if not uploaded_file:
    st.info("Upload a CSV, TSV, JSON, Parquet, or Excel file in the sidebar to begin.")
    st.stop()

try:
    df = load_uploaded_file(uploaded_file)
except Exception as exc:
    st.error(f"Could not load file: {exc}")
    st.stop()

st.subheader("Uploaded data preview")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"{len(df):,} rows × {len(df.columns)} columns")

if run:
    if not api_key:
        st.error("OpenRouter API key is required.")
        st.stop()

    st.subheader("Agent progress")
    log_box = st.empty()
    tool_calls_placeholder = st.empty()
    logs: list[str] = []
    tool_calls_live: list[dict] = []

    def render_tool_calls() -> None:
        with tool_calls_placeholder.container():
            st.markdown("**Tool calls**")
            if not tool_calls_live:
                st.caption("Waiting for tool calls...")
                return
            for index, tool_call in enumerate(tool_calls_live, start=1):
                running = tool_call.get("phase") == "start"
                label = f"{index}. {tool_call.get('purpose', 'run_code')}"
                if running:
                    label += " (running...)"
                with st.expander(label, expanded=running):
                    st.markdown(f"**Tool:** `{tool_call.get('tool', 'run_code')}`")
                    st.code(tool_call.get("code", ""), language="python")
                    if tool_call.get("output") is not None:
                        st.markdown("**Output**")
                        st.code(tool_call.get("output", ""))

    def on_status(message: str) -> None:
        logs.append(message)
        log_box.code("\n".join(logs))

    def on_tool_call(event: dict) -> None:
        if event.get("phase") == "start":
            tool_calls_live.append(dict(event))
        elif tool_calls_live:
            tool_calls_live[-1].update(event)
        render_tool_calls()

    render_tool_calls()

    with st.spinner("Running auto-analyst agent..."):
        try:
            result = analyze_dataframe(
                df,
                model,
                api_key,
                max_upload_rows=int(max_upload_rows),
                random_state=int(random_state),
                max_tool_rounds=int(max_tool_rounds),
                max_model_tries=int(max_model_tries),
                on_status=on_status,
                on_tool_call=on_tool_call,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    render_tool_calls()

    st.success("Analysis complete.")

    if result["sample_rows"] < result["total_rows"]:
        st.info(
            f"Agent explored a random sample of {result['sample_rows']:,} rows "
            f"out of {result['total_rows']:,} total."
        )

    with st.expander("Dataset profile", expanded=False):
        st.code(result["profile"])

    with st.expander("Sample used by agent", expanded=False):
        st.dataframe(result["sample_df"], use_container_width=True)

    st.subheader("Analysis")
    st.markdown(result["analysis"])

    st.download_button(
        "Download analysis.md",
        data=result["analysis"] + "\n",
        file_name="analysis.md",
        mime="text/markdown",
        use_container_width=True,
    )
