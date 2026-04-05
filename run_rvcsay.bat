@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python RVCsay_gui.py
