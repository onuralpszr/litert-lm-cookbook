"""
Example 11: LiteRT-LM local API server — OpenAI Responses API & Gemini API
===========================================================================
IMPORTANT: the 'serve' command needs the model in its local store.
Use 'litert-lm import', NOT 'litert-lm run', to register the model first.

Step 1 — import the model (one-time):
  litert-lm import \
    --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
    gemma-4-E2B-it.litertlm

Step 2a — start with OpenAI Responses API (endpoint: POST /v1/responses):
  litert-lm serve --api openai --host localhost --port 9379

Step 2b — start with Gemini API (endpoint: POST /v1beta/models/{id}:generateContent):
  litert-lm serve --api gemini --host localhost --port 9379

Step 3 — run this script:
  python 11_openai_api_server.py
"""

import json

import requests
from openai import OpenAI

HOST = "http://localhost:9379"
MODEL_ID = "gemma-4-E2B-it.litertlm"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. OpenAI Responses API  (litert-lm serve --api openai)
#   Endpoint: POST /v1/responses
#   Body fields: model (str), input (str), stream (bool)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# base_url must end with /v1 so the SDK resolves /v1/responses correctly
openai_client = OpenAI(
    base_url=f"{HOST}/v1",
    api_key="not-needed",  # litert-lm server requires no key
)


def openai_basic() -> None:
    """Non-streaming call via openai SDK Responses API."""
    print("=== A1. OpenAI Responses API — basic ===")
    response = openai_client.responses.create(
        model=MODEL_ID,
        input="What is the capital of Turkey?",
    )
    print(response.output_text)
    print()


def openai_streaming() -> None:
    """Streaming call via raw SSE — SDK .stream() is too strict for litert-lm's event format."""
    print("=== A2. OpenAI Responses API — streaming ===")
    payload = {
        "model": MODEL_ID,
        "input": "Write a short haiku about the Black Sea.",
        "stream": True,
    }
    event_type = None
    with requests.post(f"{HOST}/v1/responses", json=payload, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                event_type = None  # blank line resets the SSE message
                continue
            text = line.decode("utf-8")
            if text.startswith("event: "):
                event_type = text[7:]
            elif text.startswith("data: "):
                raw = text[6:]
                if raw == "[DONE]":
                    break
                if event_type == "response.output_text.delta":
                    try:
                        chunk = json.loads(raw)
                        delta = chunk.get("delta", {}).get("text", "")
                        if delta:
                            print(delta, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
    print("\n")


def openai_raw_http_basic() -> None:
    """Non-streaming POST /v1/responses with raw requests (no SDK)."""
    print("=== A3. OpenAI Responses API — raw HTTP POST ===")
    payload = {
        "model": MODEL_ID,
        "input": "Explain on-device AI in one sentence.",
    }
    resp = requests.post(f"{HOST}/v1/responses", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    text = data["output"][0]["content"][0]["text"]
    print(text)
    print()


def openai_raw_http_streaming() -> None:
    """Streaming POST /v1/responses with raw SSE parsing."""
    print("=== A4. OpenAI Responses API — raw HTTP streaming ===")
    payload = {
        "model": MODEL_ID,
        "input": "Name three things Turkey is famous for.",
        "stream": True,
    }
    with requests.post(f"{HOST}/v1/responses", json=payload, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            # SSE lines: "event: response.output_text.delta\ndata: {...}"
            if text.startswith("data: "):
                raw = text[6:]
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    delta = chunk.get("delta", {}).get("text", "")
                    if delta:
                        print(delta, end="", flush=True)
                except json.JSONDecodeError:
                    pass
    print("\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. Gemini API  (litert-lm serve --api gemini)
#   Non-streaming: POST /v1beta/models/{model_id}:generateContent
#   Streaming:     POST /v1beta/models/{model_id}:streamGenerateContent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GEMINI_BASE = f"{HOST}/v1beta/models/{MODEL_ID}"


def gemini_basic() -> None:
    """Non-streaming Gemini generateContent."""
    print("=== B1. Gemini API — basic ===")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "What is the capital of Turkey?"}]}]
    }
    resp = requests.post(f"{GEMINI_BASE}:generateContent", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print(data["candidates"][0]["content"]["parts"][0]["text"])
    print()


def gemini_with_system_instruction() -> None:
    """Non-streaming with system instruction."""
    print("=== B2. Gemini API — system instruction ===")
    payload = {
        "systemInstruction": {
            "parts": [
                {"text": "You are a concise geography expert. Keep answers under two sentences."}
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": "Tell me about Trabzon."}]}],
    }
    resp = requests.post(f"{GEMINI_BASE}:generateContent", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print(data["candidates"][0]["content"]["parts"][0]["text"])
    print()


def gemini_multi_turn() -> None:
    """Multi-turn conversation via Gemini API (history in contents)."""
    print("=== B3. Gemini API — multi-turn ===")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Name one city on the Black Sea coast of Turkey."}],
            },
            {
                "role": "model",
                "parts": [{"text": "Trabzon is a major city on the Black Sea coast."}],
            },
            {"role": "user", "parts": [{"text": "What is that city known for?"}]},
        ]
    }
    resp = requests.post(f"{GEMINI_BASE}:generateContent", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print(data["candidates"][0]["content"]["parts"][0]["text"])
    print()


def gemini_streaming() -> None:
    """Streaming via Gemini streamGenerateContent (SSE)."""
    print("=== B4. Gemini API — streaming ===")
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "Write a short poem about on-device AI."}]}
        ]
    }
    with requests.post(
        f"{GEMINI_BASE}:streamGenerateContent",
        json=payload,
        stream=True,
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                raw = text[6:]
                try:
                    chunk = json.loads(raw)
                    for part in (
                        chunk.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    ):
                        t = part.get("text", "")
                        if t:
                            print(t, end="", flush=True)
                except json.JSONDecodeError:
                    pass
    print("\n")


if __name__ == "__main__":
    openai_basic()
    openai_streaming()
    openai_raw_http_basic()
    openai_raw_http_streaming()

    # emini API (requires: litert-lm serve --api gemini)
    # gemini_basic()
    # gemini_with_system_instruction()
    # gemini_multi_turn()
    # gemini_streaming()
