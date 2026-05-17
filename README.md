# 🍳 LiteRT-LM Cookbook

> Run Gemma-4 on-device with Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) no cloud, no API key, no network required.

[![License](https://img.shields.io/github/license/onuralpszr/litert-llm-cookbook)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=fff)](https://www.python.org/)
[![LiteRT-LM](https://img.shields.io/badge/LiteRT--LM-nightly-4285F4?logo=google&logoColor=fff)](https://github.com/google-ai-edge/LiteRT-LM)
[![Model](https://img.shields.io/badge/Model-Gemma--4%20E4B%20Instruct-orange?logo=google&logoColor=fff)](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/01_basic_chat.ipynb)

A collection of runnable Python scripts and Google Colab notebooks that show how to use LiteRT-LM to run Gemma-4 completely on-device. Every example is self-contained and written to be read in order — from the simplest single-turn chat all the way up to a full local OpenAI-compatible API server.

The model used throughout is **Gemma-4 E4B Instruct** (4 billion parameters, edge-optimised), a strong balance between capability and speed on consumer hardware.

---

## 🤔 What is LiteRT-LM?

LiteRT-LM is Google's runtime for running large language models locally on CPU and GPU without a network connection. It exposes a Python API, a command-line tool (`litert-lm`), and a local server that speaks the OpenAI Responses API and the Gemini API. All inference stays on your machine.

---

## 📁 Repository layout

```
litert-llm-cookbook/
  python/            Python scripts (01 through 11)
  colab/             Google Colab notebooks covering the same examples
  requirements.txt   Python dependencies
  README.md          This file
```

---

## ✅ Prerequisites

- Python 3.10 or newer
- `pip` or `uv`
- For GPU examples (04, 05, 10): a GPU with a compatible driver
- For the API server example (11): the `litert-lm` CLI must be on your PATH

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv pip install -r requirements.txt
```

The `requirements.txt` pulls in `litert-lm-api-nightly`, `litert-lm`, `openai`, and `requests`.

---

## 📥 Downloading the model

Scripts 01 through 10 expect the model file to sit **in the same directory where you run the script**. Script 11 uses the model registered in the LiteRT-LM local store instead.

### Option A — Direct download

Download the file from Hugging Face and place it next to the scripts:

```bash
curl -L \
  "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm?download=true" \
  -o gemma-4-E4B-it.litertlm
```

Or open the URL directly in a browser:

```
https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm?download=true
```

The file is several gigabytes, so the download takes a few minutes on a typical connection.

### Option B — LiteRT-LM CLI (required for example 11)

The `litert-lm import` command downloads the model and registers it in the local model store (`~/.litert-lm/models/`). The API server in example 11 needs the model registered this way.

```bash
litert-lm import \
    --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm \
    gemma-4-E4B-it.litertlm
```

Expected output:

```
Downloading gemma-4-E4B-it.litertlm from litert-community/gemma-4-E4B-it-litert-lm...
Successfully imported model to /Users/you/.litert-lm/models/gemma-4-E4B-it.litertlm/model.litertlm
You can now run the model with 'litert-lm run gemma-4-E4B-it.litertlm'
```

---

## 🚀 Quick start

After placing the model file next to the scripts, run any example directly:

```bash
cd python
python 01_basic_chat.py
```

---

## 📋 Examples at a glance

| # | What it shows | Python | Colab |
|---|---------------|--------|-------|
| 01 | Single synchronous request and response | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/01_basic_chat.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/01_basic_chat.ipynb) |
| 02 | Interactive terminal chat with streaming output | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/02_streaming_chat_loop.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/02_streaming_chat_loop.ipynb) |
| 03 | Setting a persona through a system message | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/03_system_prompt.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/03_system_prompt.ipynb) |
| 04 | Running inference on GPU instead of CPU | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/04_gpu_backend.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/04_gpu_backend.ipynb) |
| 05 | GPU with multi-token prediction for faster output | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/05_speculative_decoding.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/05_speculative_decoding.ipynb) |
| 06 | Registering Python functions as callable tools | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/06_tool_use.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/06_tool_use.ipynb) |
| 07 | Sending an audio file alongside a text prompt | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/07_multimodal_audio.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/07_multimodal_audio.ipynb) |
| 08 | Sending an image alongside a text prompt | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/08_multimodal_vision.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/08_multimodal_vision.ipynb) |
| 09 | Streaming output combined with a system persona | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/09_streaming_with_system_prompt.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/09_streaming_with_system_prompt.ipynb) |
| 10 | GPU, speculative decoding, tools, and streaming together | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/10_all_features.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/10_all_features.ipynb) |
| 11 | Local OpenAI and Gemini API server | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](python/11_openai_api_server.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/11_openai_api_server.ipynb) |

---

## 🌐 API server (example 11 in detail)

Example 11 demonstrates two server modes. Both require the model to be imported first (Option B above).

**Start the OpenAI Responses API server:**

```bash
litert-lm serve --api openai --host localhost --port 9379
```

**Start the Gemini API server:**

```bash
litert-lm serve --api gemini --host localhost --port 9379
```

Then run the client script in a separate terminal while the server is running:

```bash
python 11_openai_api_server.py
```

The script covers four usage patterns: the OpenAI Python SDK, raw SSE streaming, Gemini basic chat, and Gemini multi-turn conversation.

---

## 🪪 License

See [LICENSE](LICENSE).
