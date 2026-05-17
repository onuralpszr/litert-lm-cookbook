# Google Colab Notebooks

This folder contains Google Colab notebooks that mirror the Python scripts in the `python/` folder. Each notebook is a single file you can open directly in Colab, run cell by cell, and modify without any local setup.

---

## How Colab is different from running locally

When you run on Colab the model file needs to come from somewhere. A local `.litertlm` file sitting on your laptop is not accessible inside a Colab runtime. The notebooks handle this by downloading the model from Hugging Face at the start of each session. The download runs once per runtime and the file is stored at `/content/gemma-4-E2B-it.litertlm`. Every code cell that references `MODEL_PATH` points to that path.

If you connect to a Colab runtime with a GPU, examples 04, 05, and 10 will use it automatically. Free-tier runtimes usually offer a T4. If you have Colab Pro you can request an A100 or L4 for faster generation.

---

## First-time setup (included in each notebook)

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
    "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm?download=true",
    "-o", "/content/gemma-4-E2B-it.litertlm"
], check=True)
```

After those two cells finish you can run the rest of the notebook in any order.

---

## Notebooks

### 01 - Basic Chat

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/01_basic_chat.ipynb)

Mirrors `01_basic_chat.py`. Sends a single message and prints the full response. Good for a first sanity check that the model loaded correctly and is generating text.

---

### 02 - Streaming Chat Loop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/02_streaming_chat_loop.ipynb)

Mirrors `02_streaming_chat_loop.py`. Because Colab does not have an interactive terminal loop, this notebook uses a fixed list of questions and streams the answers one by one, printing tokens as they arrive. You can edit the question list in the cell to try your own prompts.

---

### 03 - System Prompt

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/03_system_prompt.ipynb)

Mirrors `03_system_prompt.py`. Shows how to set a system persona before the first user message. The notebook uses the concise Python expert persona from the script and asks two follow-up questions to demonstrate that the persona carries across turns.

---

### 04 - GPU Backend

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/04_gpu_backend.ipynb)

Mirrors `04_gpu_backend.py`. Initialises the engine with the GPU backend. Make sure to select a GPU runtime in Colab before running this notebook. Go to Runtime, then Change runtime type, and pick T4 GPU or any other available GPU option.

---

### 05 - Speculative Decoding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/05_speculative_decoding.ipynb)

Mirrors `05_speculative_decoding.py`. Adds multi-token prediction on top of the GPU backend. The notebook includes a timing cell before and after so you can see the difference in generation speed. Requires a GPU runtime.

---

### 06 - Tool Use

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/06_tool_use.ipynb)

Mirrors `06_tool_use.py`. Defines Python functions as tools and passes them to the model. The model calls them automatically when a question requires computation or a lookup.

---

### 07 - Multimodal Audio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/07_multimodal_audio.ipynb)

Mirrors `07_multimodal_audio.py`. Includes a cell that generates a short test `.wav` file using the Python standard library so you do not need to upload your own audio. Swap the generated file for a real recording to test transcription or summarisation.

---

### 08 - Multimodal Vision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/08_multimodal_vision.ipynb)

Mirrors `08_multimodal_vision.py`. Includes a cell that downloads a small test image from the web so you can run the notebook without uploading anything. Replace the image path to use your own photo or screenshot.

---

### 09 - Streaming with System Prompt

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/09_streaming_with_system_prompt.ipynb)

Mirrors `09_streaming_with_system_prompt.py`. Sets a senior software engineer persona and streams the answer to a question about on-device AI. Tokens are flushed to the cell output as they arrive, so you see the response build up in real time.

---

### 10 - All Features Combined

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/10_all_features.ipynb)

Mirrors `10_all_features.py`. The kitchen-sink demo: GPU, speculative decoding, tools, and streaming in one notebook. Three queries run in sequence. Two call tools and one is free-form. Requires a GPU runtime. A good template to copy when building something production-like.

---

### 11 - OpenAI and Gemini API Server

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onuralpszr/litert-llm-cookbook/blob/main/colab/11_openai_api_server.ipynb)

Mirrors `11_openai_api_server.py`. Running a local server inside Colab requires a bit more setup than the other notebooks. The server runs in a background subprocess and the client cells connect to it over localhost.

The notebook does the following in order:

1. Installs dependencies and downloads the model.
2. Imports the model into the LiteRT-LM local store:

```python
!litert-lm import \
    --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
    gemma-4-E2B-it.litertlm
```

3. Starts the server in a background subprocess:

```python
import subprocess, time
server = subprocess.Popen(
    ["litert-lm", "serve", "--api", "openai", "--host", "localhost", "--port", "9379"]
)
time.sleep(5)
```

4. Runs the four OpenAI API patterns and four Gemini API patterns from the Python script. Each pattern is its own cell so you can run just the one you are interested in.

5. A final cell shuts the server down cleanly:

```python
server.terminate()
```

---

## Tips for working in Colab

The runtime is recycled after 90 minutes of inactivity on the free tier, which means the downloaded model and installed packages are gone. If you are doing extended work, consider saving the `.litertlm` file to Google Drive and mounting Drive at the start of each session to avoid re-downloading.

```python
from google.colab import drive
drive.mount('/content/drive')
MODEL_PATH = "/content/drive/MyDrive/models/gemma-4-E2B-it.litertlm"
```

That way the multi-gigabyte download only happens once and subsequent sessions load from Drive in a fraction of the time.
