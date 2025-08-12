@echo off
set PORT=%1
if "%PORT%"=="" set PORT=8000

echo Starting Albanian Legal RAG API on port %PORT% ...
uvicorn api:app --host 0.0.0.0 --port %PORT% --reload
