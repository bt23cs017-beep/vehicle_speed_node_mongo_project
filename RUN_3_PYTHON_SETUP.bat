@echo off
cd /d "%~dp0python_cv"
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Python CV environment ready.
pause
