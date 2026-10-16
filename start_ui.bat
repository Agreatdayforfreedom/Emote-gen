@echo off
rem Starts the Sprite Animator web app (opens in your browser). ComfyUI is started on the first generation.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m emote ui
pause
