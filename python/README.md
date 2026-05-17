# 🐍 Python Examples

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=fff)](https://www.python.org/)
[![LiteRT-LM](https://img.shields.io/badge/LiteRT--LM-nightly-4285F4?logo=google&logoColor=fff)](https://github.com/google-ai-edge/LiteRT-LM)
[![Model](https://img.shields.io/badge/Model-Gemma--4%20E4B%20Instruct-orange?logo=google&logoColor=fff)](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm)

Eleven standalone Python scripts that walk through the LiteRT-LM Python API step by step. Each script imports only the packages it needs and is short enough to read in a couple of minutes.

---

## ⚙️ Setup

Install the dependencies from the project root:

```bash
pip install -r ../requirements.txt
```

Or with `uv`:

```bash
uv pip install -r ../requirements.txt
```

---

## 📥 Getting the model

Scripts 01 through 10 load the model from a file in **the same directory where you run the script**.

**Download with curl:**

```bash
curl -L \
  "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm?download=true" \
  -o gemma-4-E4B-it.litertlm
```

**Or open the link directly in your browser:**

```
https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm?download=true
```

The file is a few gigabytes. Once the download finishes, all scripts that reference `MODEL_PATH = "gemma-4-E4B-it.litertlm"` will find it automatically.

> **Script 11 is different.** It talks to a running local server that looks up models from the LiteRT-LM local store. See the [Example 11](#11_openai_api_serverpy) section below for those setup steps.

---

## ▶️ Running an example

```bash
python 01_basic_chat.py
```

Every script suppresses verbose logs with `litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)` so your terminal stays readable.

---

## 📖 Script-by-script guide

### 01_basic_chat.py

The most minimal example. Opens an engine, creates a conversation, sends one message, and prints the full response. Good for a first sanity check that the model loaded correctly.

```bash
python 01_basic_chat.py
```

---

### 02_streaming_chat_loop.py

An interactive chat loop that streams tokens to the terminal as they are generated. Type anything and press Enter to get a response. Type `exit` or `quit` to stop.

```bash
python 02_streaming_chat_loop.py
```

---

### 03_system_prompt.py

Shows how to seed the conversation with a system message before the first user turn. The model is told to act as a concise Python expert. Two questions are asked back-to-back to show that the persona persists across turns.

```bash
python 03_system_prompt.py
```

---

### 04_gpu_backend.py

Same as example 01 but the engine is initialised with `backend=litert_lm.Backend.GPU`. A `cache_dir` is also specified so compiled GPU kernels are stored between runs. Requires a compatible GPU and the GPU-enabled build of LiteRT-LM.

```bash
python 04_gpu_backend.py
```

---

### 05_speculative_decoding.py

Adds `enable_speculative_decoding=True` on top of the GPU backend. Multi-token prediction lets the model draft several tokens at once and verify them in parallel, which can noticeably cut generation time. Requires GPU.

```bash
python 05_speculative_decoding.py
```

---

### 06_tool_use.py

Defines six Python functions across five categories (arithmetic, unit conversion, weather, country info, BMI) and passes them as tools. The model calls them automatically when a question requires computation or a data lookup. Each category runs in its own focused conversation so the model only sees the tools it needs.

```bash
python 06_tool_use.py
```

---

### 07_multimodal_audio.py

Demonstrates how to send an audio file together with a text prompt. The user message is a list with an `audio` content block pointing to a `.wav` file and a `text` block with the instruction. Update `AUDIO_FILE` to point to a real audio file before running.

```bash
python 07_multimodal_audio.py
```

---

### 08_multimodal_vision.py

Same idea as example 07, but with an image. The user message contains an `image` content block and a `text` block asking the model to describe what it sees. Update `IMAGE_FILE` to point to a real image before running.

```bash
python 08_multimodal_vision.py
```

---

### 09_streaming_with_system_prompt.py

Combines a system persona (a senior software engineer specialising in on-device AI) with streaming output. The response is printed token by token. A good template for applications that need both a custom persona and a responsive UI.

```bash
python 09_streaming_with_system_prompt.py
```

---

### 10_all_features.py

The kitchen-sink demo: GPU, speculative decoding, tools, and streaming all at once. Three queries run in sequence: two exercise tools and one is free-form. A good reference for a production-like setup.

```bash
python 10_all_features.py
```

---

### 11_openai_api_server.py

This script connects to a local LiteRT-LM server over HTTP rather than loading a model file directly. Two server modes are supported; pick one before running.

**Step 1: import the model into the local store (one-time only):**

```bash
litert-lm import \
    --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm \
    gemma-4-E4B-it.litertlm
```

The model ends up at `~/.litert-lm/models/gemma-4-E4B-it.litertlm/model.litertlm`. This step only needs to run once.

**Step 2a: start the OpenAI Responses API server:**

```bash
litert-lm serve --api openai --host localhost --port 9379
```

**Step 2b: or start the Gemini API server:**

```bash
litert-lm serve --api gemini --host localhost --port 9379
```

**Step 3: run the client in a separate terminal:**

```bash
python 11_openai_api_server.py
```

The script contains four sections under the OpenAI API (`A1`–`A4`) and four under the Gemini API (`B1`–`B4`). The Gemini ones are commented out by default because only one server can be active at a time.

| Section | Pattern |
|---------|---------|
| A1 | Plain OpenAI SDK call |
| A2 | Streaming via raw SSE |
| A3 | Raw HTTP POST without the SDK |
| A4 | Raw HTTP streaming |
| B1 | Basic generateContent |
| B2 | generateContent with a system instruction |
| B3 | Multi-turn conversation |
| B4 | Streaming via streamGenerateContent |

---

## 💡 Notes on GPU examples

GPU examples (04, 05, 10) require a GPU that LiteRT-LM supports. On first run the engine compiles and caches GPU kernels, so startup takes longer than usual. Subsequent runs reuse the cache and start up faster.

If you do not have a GPU, change `backend=litert_lm.Backend.GPU` to `backend=litert_lm.Backend.CPU` and remove `enable_speculative_decoding=True`. Everything else works the same on CPU.

---

## 📚 Citation

If you use this cookbook in your research or work, please cite it as:

```bibtex
@misc{litert-llm-cookbook,
  author       = {Onuralp Sezer},
  title        = {LiteRT-LM Cookbook},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/onuralpszr/litert-llm-cookbook}},
}
```

This project builds on the following resources. Please also cite them if they are relevant to your work:

- **LiteRT-LM** [github.com/google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
- **Gemma** [ai.google.dev/gemma](https://ai.google.dev/gemma)
- **litert-community on Hugging Face** [huggingface.co/litert-community](https://huggingface.co/litert-community)

---

## ⚠️ Disclaimer

This is an independent community project and is not affiliated with, endorsed by, or sponsored by Google. LiteRT-LM and Gemma are trademarks of Google LLC.
