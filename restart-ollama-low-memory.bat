@echo off
echo Stopping any running Ollama processes...
taskkill /F /IM ollama.exe 2>nul

echo Setting low memory environment variables...
set OLLAMA_MODELS=%USERPROFILE%\.ollama\models
set OLLAMA_NUM_GPU=0
set OLLAMA_NUM_THREAD=1
set OLLAMA_CONTEXT_LENGTH=512
set OLLAMA_LOW_VRAM=true
set OLLAMA_F16_KV=true

echo Starting Ollama with low memory settings...
start /b ollama serve

echo Done! Ollama is now running with low memory settings.
echo Wait a moment for the server to start, then try your query again.
echo.
pause 