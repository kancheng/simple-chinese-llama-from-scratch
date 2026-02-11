# -*- coding: utf-8 -*-
"""
SimpleLLM Demo — 前端展示頁
請先啟動 server.py（API 埠 8000），再執行本檔並開啟瀏覽器。
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="SimpleLLM Demo")

# 預設 API 位址（與 main.py 一致）
DEFAULT_API = "http://localhost:8000"

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SimpleLLM Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #0f0f12;
      --surface: #18181c;
      --border: #2a2a32;
      --text: #e8e6e3;
      --muted: #8b8b92;
      --accent: #7c9cf5;
      --accent-hover: #9ab0f7;
      --success: #7dd3a0;
      --error: #e07c7c;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Noto Sans TC', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem;
      line-height: 1.6;
    }
    .container {
      max-width: 640px;
      margin: 0 auto;
    }
    h1 {
      font-size: 1.75rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, var(--accent), #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .sub {
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 2rem;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 0.5rem;
    }
    input[type="text"], input[type="number"] {
      width: 100%;
      padding: 0.75rem 1rem;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.9rem;
    }
    input:focus {
      outline: none;
      border-color: var(--accent);
    }
    .row {
      display: flex;
      gap: 1rem;
      align-items: flex-end;
      flex-wrap: wrap;
    }
    .row .field { flex: 1; min-width: 140px; }
    .row .field.num { max-width: 100px; }
    button {
      font-family: inherit;
      font-weight: 600;
      padding: 0.75rem 1.5rem;
      background: var(--accent);
      color: var(--bg);
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.95rem;
      white-space: nowrap;
    }
    button:hover { background: var(--accent-hover); }
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .output {
      margin-top: 1rem;
      padding: 1rem;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      font-family: 'JetBrains Mono', 'Noto Sans TC', monospace;
      font-size: 0.95rem;
      white-space: pre-wrap;
      word-break: break-all;
      min-height: 80px;
    }
    .output.placeholder { color: var(--muted); }
    .output.success { border-color: var(--success); }
    .output.error { border-color: var(--error); color: var(--error); }
    .status {
      font-size: 0.85rem;
      color: var(--muted);
      margin-top: 0.5rem;
    }
    .batch-note {
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 0.75rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>SimpleLLM Demo</h1>
    <p class="sub">西遊記風格字級語言模型 · 依序先啟動 main.py 再開此頁</p>

    <div class="card">
      <label>API 位址</label>
      <input type="text" id="apiUrl" value="http://localhost:8000" placeholder="http://localhost:8000" />
    </div>

    <div class="card">
      <div class="row">
        <div class="field num">
          <label>生成長度 (tokens)</label>
          <input type="number" id="maxTokens" value="30" min="5" max="100" />
        </div>
        <div class="field" style="flex: 0;">
          <label>&nbsp;</label>
          <button type="button" id="btnGen">生成</button>
        </div>
      </div>

      <label style="margin-top: 1rem;">輸出</label>
      <div class="output placeholder" id="output">點「生成」開始</div>
      <div class="status" id="status"></div>
      <div class="batch-note" id="batchNote"></div>
    </div>
  </div>

  <script>
    const apiUrl = document.getElementById('apiUrl');
    const maxTokens = document.getElementById('maxTokens');
    const btnGen = document.getElementById('btnGen');
    const output = document.getElementById('output');
    const status = document.getElementById('status');
    const batchNote = document.getElementById('batchNote');

    function setOutput(text, isError) {
      output.textContent = text || '';
      output.classList.remove('placeholder', 'success', 'error');
      if (isError) output.classList.add('error');
      else if (text) output.classList.add('success');
      else output.classList.add('placeholder');
    }

    function setStatus(msg) { status.textContent = msg || ''; }
    function setBatchNote(msg) { batchNote.textContent = msg || ''; }

    async function generate() {
      const base = (apiUrl.value || '').replace(/\\/$/, '');
      const url = base + '/generate/?max_new_tokens=' + (parseInt(maxTokens.value, 10) || 20);
      setOutput('生成中…');
      setStatus('');
      setBatchNote('');
      btnGen.disabled = true;

      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ idx: [[0]] })
        });
        const data = await res.json();
        if (!res.ok) {
          setOutput('錯誤: ' + (data.detail || res.statusText), true);
          setStatus('HTTP ' + res.status);
          return;
        }
        const list = Array.isArray(data) ? data : [data];
        const primary = list[0];
        const text = typeof primary === 'string' ? primary : (primary && primary.text ? primary.text : JSON.stringify(primary));
        setOutput(text);
        if (list.length > 1) setBatchNote('本次共 ' + list.length + ' 條序列，上為第一條');
        setStatus('完成');
      } catch (e) {
        setOutput('請求失敗: ' + e.message, true);
        setStatus('請確認 main.py 已啟動且埠為 8000，並已設定 CORS');
      } finally {
        btnGen.disabled = false;
      }
    }

    btnGen.addEventListener('click', generate);
  </script>
</body>
</html>
"""


@app.get("/generate")
@app.get("/generate/")
def generate_redirect():
    """5500 是 Demo 前端，生成 API 在 8000。回傳說明並建議用首頁或直接打 API。"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=200,
        content={
            "message": "此為 Demo 前端（port 5500）。生成 API 在 port 8000。",
            "usage": "請開啟首頁 http://localhost:5500/ 使用「生成」按鈕，或對 http://localhost:8000/generate/ 發 POST 請求。",
            "api_url": "http://localhost:8000/generate/",
        },
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn
    print("Demo 前端啟動中 … 請在瀏覽器開啟: http://localhost:5500/ 或 http://127.0.0.1:5500/")
    print("（勿使用 0.0.0.0，瀏覽器無法連線）")
    uvicorn.run(app, host="0.0.0.0", port=5500)
