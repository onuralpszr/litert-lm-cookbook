# 📓 Google Colab Notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/01_basic_chat.ipynb)
[![LiteRT-LM](https://img.shields.io/badge/LiteRT--LM-nightly-4285F4?logo=google&logoColor=fff)](https://github.com/google-ai-edge/LiteRT-LM)
[![Model](https://img.shields.io/badge/Model-Gemma--4%20E4B%20Instruct-orange?logo=google&logoColor=fff)](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm)

Google Colab notebooks that mirror the Python scripts in the `python/` folder. Each notebook is a single file you can open directly in Colab, run cell by cell, and modify without any local setup — no model download, no environment configuration, just click and run.

---

## 📋 Notebooks

| # | What it shows | Colab |
|---|---------------|-------|
| 01 | Single synchronous request and response | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/01_basic_chat.ipynb) |
| 02 | Interactive chat loop with streaming output | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/02_streaming_chat_loop.ipynb) |
| 03 | Setting a persona through a system message | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/03_system_prompt.ipynb) |
| 04 | Running inference on GPU instead of CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/04_gpu_backend.ipynb) |
| 05 | GPU with multi-token prediction for faster output | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/05_speculative_decoding.ipynb) |
| 06 | Registering Python functions as callable tools | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/06_tool_use.ipynb) |
| 07 | Sending an audio file alongside a text prompt | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/07_multimodal_audio.ipynb) |
| 08 | Sending an image alongside a text prompt | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/08_multimodal_vision.ipynb) |
| 09 | Streaming output combined with a system persona | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/09_streaming_with_system_prompt.ipynb) |
| 10 | GPU, speculative decoding, tools, and streaming together | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/10_all_features.ipynb) |
| 11 | Local OpenAI and Gemini API server | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/11_openai_api_server.ipynb) |

---

## 🆚 How Colab differs from running locally

When you run on Colab the model file needs to come from somewhere — a local `.litertlm` file on your laptop is not accessible inside a Colab runtime. The notebooks handle this by downloading the model from Hugging Face at the start of each session. The download runs once per runtime and the file is stored at `/content/gemma-4-E4B-it.litertlm`. Every cell that references `MODEL_PATH` points to that path.

If you connect to a Colab runtime with a GPU, examples 04, 05, and 10 will use it automatically. Free-tier runtimes usually offer a T4. If you have Colab Pro you can request an A100 or L4 for faster generation.

---

## 🛠️ First-time setup (included in each notebook)

Every notebook starts with two setup cells.

**Install dependencies:**

```python
!pip install litert-lm-api-nightly litert-lm openai requests -q
```

**Download the model:**

```python
import subprocess
subprocess.run([
    "curl", "-L",
    "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm?download=true",
    "-o", "/content/gemma-4-E4B-it.litertlm"
], check=True)
```

After those two cells finish you can run the rest of the notebook in any order.

---

## 📝 Notebook details

### 01 - Basic Chat

Sends a single message and prints the full response. Good for a first sanity check that the model loaded correctly and is generating text.

### 02 - Streaming Chat Loop

Because Colab does not have an interactive terminal loop, this notebook uses a fixed list of questions and streams the answers one by one, printing tokens as they arrive. Edit the question list to try your own prompts.

### 03 - System Prompt

Shows how to set a system persona before the first user message. The notebook uses a concise Python expert persona and asks two follow-up questions to demonstrate that the persona carries across turns.

### 04 - GPU Backend

Initialises the engine with the GPU backend. Make sure to select a GPU runtime in Colab before running: go to **Runtime**, then **Change runtime type**, and pick **T4 GPU** or any available GPU option.

### 05 - Speculative Decoding

Adds multi-token prediction on top of the GPU backend. The notebook includes a timing cell before and after so you can see the difference in generation speed. Requires a GPU runtime.

### 06 - Tool Use

Defines six Python functions across five categories (arithmetic, unit conversion, weather, country info, BMI) and passes them as tools. Each category runs in its own focused conversation so the model only sees the tools it needs. The model calls them automatically when a question requires a lookup or calculation.

### 07 - Multimodal Audio

Includes a cell that generates a short test `.wav` file using the Python standard library so you do not need to upload your own audio. Swap the generated file for a real recording to test transcription or summarisation.

### 08 - Multimodal Vision

Includes a cell that downloads a small test image from the web so you can run the notebook without uploading anything. Replace the image path to use your own photo or screenshot.

### 09 - Streaming with System Prompt

Sets a senior software engineer persona and streams the answer to a question about on-device AI. Tokens are flushed to the cell output as they arrive, so you see the response build up in real time.

### 10 - All Features Combined

The kitchen-sink demo: GPU, speculative decoding, tools, and streaming in one notebook. Three queries run in sequence — two call tools and one is free-form. Requires a GPU runtime.

### 11 - OpenAI and Gemini API Server

The server runs in a background subprocess and client cells connect to it over localhost. The notebook imports the model into the LiteRT-LM local store, starts the server, runs all eight API patterns (four OpenAI, four Gemini), then shuts the server down in a final cell.

---

## 💡 Tips for working in Colab

The runtime is recycled after 90 minutes of inactivity on the free tier, which means the downloaded model and installed packages are gone. If you are doing extended work, consider saving the `.litertlm` file to Google Drive and mounting Drive at the start of each session to avoid re-downloading.

```python
from google.colab import drive
drive.mount('/content/drive')
MODEL_PATH = "/content/drive/MyDrive/models/gemma-4-E4B-it.litertlm"
```

That way the multi-gigabyte download only happens once and subsequent sessions load from Drive in a fraction of the time.
