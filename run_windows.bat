@echo off
REM One-click setup + run for Windows. First run installs packages (needs internet).
cd /d "%~dp0"
if not exist .venv (
    python -m venv .venv || (echo Python 3.10+ not found. Install from python.org and tick "Add to PATH". & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
python -m streamlit run app.py
pause
