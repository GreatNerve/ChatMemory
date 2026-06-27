# ChatMemory API



FastAPI backend — see repo `docs/` for full design.



## Setup



```bash

cd backend

cp .env.example .env

# Set GEMINI_API_KEY in .env

uv sync

```



### Windows: numpy/torch blocked ("Application Control policy")



Defender folder exclusions are **not enough** if **Smart App Control** is On.



1. **Windows Security** → **App & browser control** → **Smart App Control** → **Off**

2. **Restart** your PC

3. In PowerShell **as Administrator**:

   ```powershell

   cd D:\@2026\RAG_TEST\backend\scripts

   .\fix-windows-ml.ps1

   ```

4. Confirm: `uv run python -c "import torch; print(torch.__version__)"`



Embeddings use **sentence-transformers** (`intfloat/multilingual-e5-large`) locally. Q&A uses **LangChain** + **Google Gemini** (Interactions API via `GEMINI_API_KEY`). Persona chat streams from `gemini.py` directly.



Get an API key: [Google AI Studio](https://aistudio.google.com/apikey)



## Run



```bash

uv run uvicorn app.main:app --reload --port 8000

```



API base: `http://127.0.0.1:8000/api/v1`



## Tests



```bash

uv run pytest tests/unit -q

uv run pytest -q   # full suite; requires numpy/torch allowed by OS policy

```



On Windows, if pytest fails with *Application Control policy* blocking numpy DLLs, run `scripts/fix-windows-ml.ps1` after disabling Smart App Control.

