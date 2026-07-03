@echo off
setlocal
cd /d "%~dp0"

REM Resolve a working Python (prefer `python`, fall back to `py -3`)
set "PY=python"
where python >nul 2>&1
if errorlevel 1 (
  where py >nul 2>&1
  if errorlevel 1 (
    echo [error] Python not found on PATH. Install Python 3 or add it to PATH.
    pause
    exit /b 1
  )
  set "PY=py -3"
)

echo [1/2] Starting FastAPI on http://127.0.0.1:8000 ...
REM Install deps quietly, then launch uvicorn directly; if the `uvicorn` shim is
REM missing (common when Scripts\ is not on PATH), fall back to `python -m uvicorn`.
start "classroom-api" cmd /k "%PY% -m pip install -q fastapi uvicorn python-multipart 2>nul & ( uvicorn api_server:app --host 127.0.0.1 --port 8000 || %PY% -m uvicorn api_server:app --host 127.0.0.1 --port 8000 )"

timeout /t 2 /nobreak >nul

echo [2/2] Starting Next.js on http://localhost:3000 ...
cd web
if not exist .env.local (
  echo # Browser uses /detection-api proxy to 127.0.0.1:8000 ^(no CORS^). Optional direct URL:>> .env.local
  echo # NEXT_PUBLIC_API_URL=http://127.0.0.1:8000>> .env.local
  echo # NEXT_PUBLIC_API_PROXY=false>> .env.local
)

REM Try npm first; if it fails (missing on PATH or script error), fall back to npx next.
where npm >nul 2>&1
if errorlevel 1 (
  echo [warn] npm not found on PATH, trying npx next dev ...
  npx --yes next dev
) else (
  call npm run dev
  if errorlevel 1 (
    echo [warn] npm run dev failed, falling back to npx next dev ...
    npx --yes next dev
  )
)

pause
endlocal
