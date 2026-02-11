# -*- coding: utf-8 -*-
"""
SimpleLLM API 伺服器（獨立於 main.py）
使用方式：先以 main.py 訓練並保存模型到 hf_model_save/，再執行 python server.py
"""
import json
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_arch import Llama

# 路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "hf_model_save", "config.json")
MODEL_PATH = os.path.join(BASE_DIR, "hf_model_save", "pytorch_model.bin")
DATA_PATH = os.path.join(BASE_DIR, "xiyouji.txt")

# 載入 config（與 main 保存的 config.json 一致）
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# 詞表與解碼（與 main 一致）
lines = open(DATA_PATH, "r", encoding="utf-8").read()
vocab = sorted(list(set(lines)))
itos = {i: ch for i, ch in enumerate(vocab)}


def decode(ids):
    return "".join([itos[i] for i in ids])


# 載入模型
model = Llama(CONFIG)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

# FastAPI
app = FastAPI(title="SimpleLLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    idx: list


@app.get("/generate/")
async def generate_help():
    return {
        "message": "請使用 POST 請求。例如: curl -X POST http://localhost:8000/generate/ -H \"Content-Type: application/json\" -d '{\"idx\": [[0]]}'",
        "doc": "POST /generate/，body: {\"idx\": [[token_id, ...], ...]}（可選 max_new_tokens）",
    }


@app.post("/generate/")
async def generate_api(max_new_tokens: int = 20):
    # 固定 5 個樣本、每樣本 1 個 token 作為起頭（與 main 行為一致）
    idx = torch.zeros(5, 1).long()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -CONFIG["context_window"] :])
        last_logits = logits[:, -1, :]
        p = F.softmax(last_logits, dim=-1)
        idx_next = torch.multinomial(p, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)
    return [decode(x) for x in idx.tolist()]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
