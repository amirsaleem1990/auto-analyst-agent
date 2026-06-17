#!/home/amir/github/auto-analyst-agent/.venv/bin/python3

"""
List free models on OpenRouter, and optionally call chat completions
with automatic fallback to the next free model if one is rate-limited
or temporarily unavailable.

No API key is required just to list models (the /models endpoint is public).
A key IS required to actually call chat/completions.

Usage:
    python3 openrouter_free_models.py                # just list free models
    OPENROUTER_API_KEY=sk-... python3 openrouter_free_models.py --chat "hello"
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def supports_tools(model: dict) -> bool:
    return "tools" in (model.get("supported_parameters") or [])


def get_free_models(tools_only: bool = False):
    """Return free models sorted by context length (largest first)."""
    req = urllib.request.Request(
        MODELS_URL, headers={"User-Agent": "free-model-lister/1.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)

    free_models = []
    for model in data.get("data", []):
        pricing = model.get("pricing", {})
        prompt_price = float(pricing.get("prompt") or 0)
        completion_price = float(pricing.get("completion") or 0)
        is_free = (prompt_price == 0 and completion_price == 0) or model["id"].endswith(":free")
        if not is_free:
            continue
        if tools_only and not supports_tools(model):
            continue
        free_models.append(
            {
                "id": model["id"],
                "name": model.get("name", ""),
                "context_length": model.get("context_length") or 0,
            }
        )

    free_models.sort(key=lambda m: m["context_length"], reverse=True)
    return free_models


def print_free_models(models):
    print(f"Found {len(models)} free models on OpenRouter:\n")
    for m in models:
        print(f"  {m['id']:<45} ctx={m['context_length']:,}")


def call_model(api_key, model_id, prompt, timeout=30):
    """Call one model. Returns text on success, raises on failure."""
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()

    req = urllib.request.Request(
        CHAT_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]


def chat_with_fallback(api_key, prompt, free_models, max_attempts=None):
    """
    Try free models in order. If one returns 429 (rate limited) or another
    error, move on to the next. Returns (model_id, response_text).
    """
    candidates = free_models if max_attempts is None else free_models[:max_attempts]

    for model in candidates:
        model_id = model["id"]
        try:
            print(f"-> trying {model_id} ...")
            text = call_model(api_key, model_id, prompt)
            return model_id, text
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"   {model_id} is rate-limited right now, skipping")
            else:
                print(f"   {model_id} failed ({e.code}), skipping")
            time.sleep(1)  # small backoff before trying the next one
        except urllib.error.URLError as e:
            print(f"   {model_id} network error ({e}), skipping")

    raise RuntimeError("All candidate free models failed or are rate-limited")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", help="Send this prompt with fallback across free models")
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    models = get_free_models()
    # print_free_models(models)
    
    if args.chat:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("\nSet OPENROUTER_API_KEY to use --chat")
            return
        model_id, text = chat_with_fallback(api_key, args.chat, models, args.max_attempts)
        print(f"\nResponded using: {model_id}\n")
        print(text)


if __name__ == "__main__":
    main()