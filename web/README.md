# Exam activity monitor (Next.js + shadcn/ui)

Browser UI for the same detection pipeline as `streamlit_app.py`. Inference runs in a **Python FastAPI** service (`../api_server.py`).

## Prerequisites

- Python 3 with project deps from repo root: `pip install -r requirements.txt`
- Node 20+ and npm

## 1) Start the API (repo root)

```bash
uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

Run from the folder that contains `detector.py` and model weights (repo root).

## 2) Start the web app

```bash
cd web
cp .env.example .env.local   # or create NEXT_PUBLIC_API_URL
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Windows shortcut

From repo root, double-click `run_web_stack.bat` (starts API in a new window, then `npm run dev`).

## Environment

| Variable | Meaning |
|----------|---------|
| `NEXT_PUBLIC_API_URL` | FastAPI base URL, e.g. `http://127.0.0.1:8000` |

## Features (parity with Streamlit)

- **Minimal / Advanced** interface
- **Image** upload → original + annotated + download PNG
- **Video** upload → up to 20 sampled frames with summaries
- **Live** → browser webcam, frames POSTed to API with per-session `LateralMotionTracker`
- Dashboard metrics, phone / back-facing alerts, optional desktop **Notification** when back-facing triggers
