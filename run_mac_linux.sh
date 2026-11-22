#!/usr/bin/env bash
# One-click setup + run for macOS / Linux. First run installs packages (needs internet).
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
python -m streamlit run app.py
