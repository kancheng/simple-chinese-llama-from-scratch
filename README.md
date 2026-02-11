## Simple LLaMA-style LLM from Scratch (Chinese "Journey to the West")

This project is a learning-oriented implementation of a small LLaMA‑style language model built **from scratch in PyTorch**, using the Chinese classic **“Journey to the West (西游记)”** as the training corpus.

The code in `main.py` walks through the evolution from a naive character-level language model to a more sophisticated architecture that incorporates:

- Basic embedding + MLP language model
- RMSNorm
- RoPE (Rotary Positional Embeddings)
- Multi‑head self‑attention
- SwiGLU activation
- A stacked LLaMA‑like block architecture
- Training, evaluation, sampling (`generate`), and saving to `hf_model_save/`

**API 與 Demo 已獨立**：推理用模型在 `model_arch.py`，API 在 `server.py`（port 8000），網頁 Demo 在 `app.py`（port 5500）。`main.py` 僅負責訓練與保存，方便閱讀與實驗。

---

## Features

- **Character-level modeling**
  - Builds a vocabulary directly from characters in `xiyouji.txt`
  - Custom `encode` / `decode` utilities for mapping between chars and integer IDs

- **Dataset & batching**
  - Uses the full text as a long 1D tensor of token IDs
  - Splits into train/validation/test (80% / 10% / 10%)
  - Sliding window context with configurable `context_window`
  - Random batch sampling via `get_batches(...)`

- **Model progression**
  1. `StupidModel` / `SimpleBrokenModel` – simple embedding + MLP with a few intentional issues (e.g. softmax before cross‑entropy) to illustrate pitfalls  
  2. `SimpleNotStupidModel` – fixes softmax misuse, trains more stably  
  3. `SimpleNotStupidModel_RMS` – adds `RMSNorm` to stabilize training  
  4. `RoPEMaskedAttentionHead` / `RoPEMaskedMultiheadAttention` – single‑head then multi‑head attention with RoPE  
  5. `RopeModel` – combines embedding, RMSNorm, RoPE attention, MLP head  
  6. `SwiGLU` – custom SwiGLU activation module  
  7. `LlamaBlock` and `Llama` – stacked blocks implementing a LLaMA‑like architecture

- **Training utilities**
  - `MASTER_CONFIG` dict to hold all hyperparameters
  - `evaluate_loss(...)` for periodic evaluation on train/val
  - `train(...)` loop with:
    - Adam optimizer
    - Optional LR scheduler (CosineAnnealingLR)
    - Logging of validation loss, basic ETA estimation

- **Text generation**
  - `generate(model, config, max_new_tokens)` performs autoregressive sampling
  - Starts from a small context (initially zeros) and samples the next token using multinomial sampling from the softmax distribution

- **Saving and loading**
  - Saves entire model and weights (e.g. `llama_model.pth`, `llama_model_params.pth`)
  - Hugging Face‑style directory `./hf_model_save/` with:
    - `pytorch_model.bin` – model weights
    - `config.json` – dumped `MASTER_CONFIG`
    - `optimizer.pt` / `scheduler.pt` – optimizer and scheduler state
  - Demonstrates how to load weights back and run inference again

- **獨立 API 與 Demo**
  - **server.py** – 獨立的 FastAPI 伺服器，從 `hf_model_save/` 載入模型，提供 `GET/POST /generate/`，埠 8000
  - **app.py** – Demo 前端（單頁），埠 5500，可設定生成長度並呼叫 API 顯示結果
  - **model_arch.py** – 推理用模型架構（Llama 與依賴），供 server 載入，無須執行 main.py
  - 本機瀏覽請使用 **http://localhost:5500/** 或 **http://127.0.0.1:5500/**，勿使用 `0.0.0.0`（會 ERR_ADDRESS_INVALID）

---

## Project Structure

```text
simplellm/
├─ main.py          # 資料、模型定義、訓練流程（無 API，僅訓練與保存）
├─ model_arch.py    # 推理用模型架構（Llama 等），供 server 載入
├─ server.py        # 獨立 API 伺服器（FastAPI，port 8000）
├─ app.py           # Demo 前端（單頁，port 5500）
├─ xiyouji.txt      # “Journey to the West” 語料（訓練用）
├─ hf_model_save/   # 訓練後產生：pytorch_model.bin、config.json 等
└─ requirements.txt # Python 依賴
```

---

## Requirements

Python 3.9+ is recommended.

Key dependencies (also listed in `requirements.txt`):

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `fastapi`
- `uvicorn`
- `pydantic`
- `nest_asyncio`
- `requests`

Install everything in one go:

```bash
pip install -r requirements.txt
```

---

## Getting Started

1. **Clone the repository and enter the directory**

   ```bash
   git clone <your-repo-url>.git
   cd simplellm
   ```

2. **(Optional but recommended) Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   # or on Windows (PowerShell)
   # .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the dataset is present**

   - The code expects a file named `xiyouji.txt` in the project root.
   - If it is missing, download a UTF‑8 encoded “Journey to the West” full text and save it as:

     ```text
     simplellm/xiyouji.txt
     ```

5. **Run the training script**

   ```bash
   python main.py
   ```

   This will:

   - Build the character vocabulary
   - Train several progressively more complex models
   - Train the final `Llama` model (with and without a cosine annealing scheduler)
   - Save checkpoints and configuration under `./hf_model_save/`
   - Print some generated text samples to the console

   **Note**: API 已獨立至 `server.py`，執行 `main.py` 僅會訓練與保存，不會啟動 HTTP 服務。

> **Warning**: Training is compute‑intensive. Hyperparameters such as `d_model`, `n_heads`, `n_layers`, `batch_size`, and `epochs` are stored in `MASTER_CONFIG`. You can lower them in `main.py` to speed up experiments on a CPU‑only machine.

---

## Running the API (server.py)

API 已從 `main.py` 抽離，改由 **server.py** 提供。

1. **先完成訓練並產生模型**（或已有 `./hf_model_save/` 下的權重與 `config.json`）。

2. **啟動 API 伺服器**

   ```bash
   python server.py
   ```

   預設監聽 **port 8000**（`http://0.0.0.0:8000`）。本機呼叫請用：

   ```bash
   curl -X POST "http://localhost:8000/generate/?max_new_tokens=30" -H "Content-Type: application/json" -d "{\"idx\": [[0]]}"
   ```

---

## Running the Demo frontend (app.py)

Demo 為單頁前端，會對上述 API 發 POST 請求並顯示生成結果。

1. **先啟動 API**：`python server.py`（port 8000）。

2. **啟動 Demo 前端**

   ```bash
   python app.py
   ```

   預設監聽 **port 5500**。啟動後終端會提示：

   - 請在瀏覽器開啟 **http://localhost:5500/** 或 **http://127.0.0.1:5500/**

3. **勿在瀏覽器使用 `http://0.0.0.0:5500/`**  
   `0.0.0.0` 僅為伺服器綁定用，瀏覽器會出現 `ERR_ADDRESS_INVALID`，請改用 `localhost` 或 `127.0.0.1`。

| 用途       | 指令            | 埠   |
|------------|-----------------|------|
| 訓練       | `python main.py` | —    |
| API 服務   | `python server.py` | 8000 |
| Demo 前端  | `python app.py` | 5500 |

---

## Customization

You can experiment with:

- **Model size**
  - `d_model` – embedding and hidden dimension size
  - `n_heads` – number of attention heads
  - `n_layers` – number of stacked `LlamaBlock`s

- **Training setup**
  - `batch_size`, `context_window`
  - `epochs`
  - Optimizer type and learning rate
  - Scheduler settings (CosineAnnealingLR parameters)

- **Sampling behavior**
  - `max_new_tokens` in `generate(...)`
  - Use temperature or top‑k / top‑p sampling (you can extend the code to support these)

All these hyperparameters live in the `MASTER_CONFIG` dictionary in `main.py`, so you can tweak them in one place.

---

## Disclaimer

This project is **for educational and experimental purposes only**:

- It is not optimized for speed, memory usage, or production deployment.
- The model is trained on a single literary work, so its outputs are limited and may not be coherent.
- The API (`server.py`) and Demo (`app.py`) are minimal and should be refactored before any real‑world use.

That said, it is a great starting point if you want to:

- Understand how modern decoder‑only LLMs like LLaMA are built
- Experiment with RMSNorm, RoPE, and SwiGLU
- Practice building and training language models end‑to‑end in PyTorch

