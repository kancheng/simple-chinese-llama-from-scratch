## Simple LLaMA-style LLM from Scratch (Chinese "Journey to the West")

This project is a learning-oriented implementation of a small LLaMA‑style language model built **from scratch in PyTorch**, using the Chinese classic **“Journey to the West (西游记)”** as the training corpus.

The code in `main.py` walks through the evolution from a naive character-level language model to a more sophisticated architecture that incorporates:

- Basic embedding + MLP language model
- RMSNorm
- RoPE (Rotary Positional Embeddings)
- Multi‑head self‑attention
- SwiGLU activation
- A stacked LLaMA‑like block architecture
- Training, evaluation, sampling (`generate`) and simple FastAPI deployment

The implementation is intentionally simple and monolithic (everything in `main.py`) to make it easy to read and experiment with.

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

- **FastAPI deployment example**
  - Minimal FastAPI app exposing a `/generate/` endpoint
  - Loads the trained LLaMA‑like model from disk
  - Uses the same character‑level `generate` logic to return generated text

> **Note**: The FastAPI part is adapted from a Jupyter/Colab notebook and may need some refactoring if you want to use it as a production‑grade API (e.g. function naming, request schema, GPU/CPU device handling).

---

## Project Structure

```text
simplellm/
├─ main.py          # All data, models, training and FastAPI example in one file
├─ xiyouji.txt      # “Journey to the West” corpus (Chinese), used for training
└─ requirements.txt # Python dependencies
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

> **Warning**: Training is compute‑intensive. Hyperparameters such as `d_model`, `n_heads`, `n_layers`, `batch_size`, and `epochs` are stored in `MASTER_CONFIG`. You can lower them in `main.py` to speed up experiments on a CPU‑only machine.

---

## Running the FastAPI Demo

The latter part of `main.py` contains an example FastAPI application that serves the trained model via HTTP.

To run a similar service outside of Jupyter/Colab, the rough steps are:

1. **Make sure you have trained and saved a model** (or have a pretrained checkpoint under `./hf_model_save/`).
2. **Install API requirements** (already in `requirements.txt`):

   ```bash
   pip install fastapi uvicorn pydantic nest_asyncio requests
   ```

3. **(Recommended) Extract the FastAPI section into a separate file**, e.g. `api.py`, and adapt it to:
   - Import the `Llama` class and `MASTER_CONFIG`
   - Load `pytorch_model.bin`
   - Define a proper request model (`InputData`) and a clean `/generate` endpoint
   - Run:

     ```bash
     uvicorn api:app --host 0.0.0.0 --port 8000
     ```

Then you can call the endpoint with:

```bash
curl -X POST "http://localhost:8000/generate/" -H "Content-Type: application/json" -d "{\"idx\": [[0]]}"
```

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
- The FastAPI server example is intentionally minimal and should be refactored before any real‑world use.

That said, it is a great starting point if you want to:

- Understand how modern decoder‑only LLMs like LLaMA are built
- Experiment with RMSNorm, RoPE, and SwiGLU
- Practice building and training language models end‑to‑end in PyTorch

