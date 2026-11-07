@echo off
cd /d "%~dp0"
call python_cv\venv\Scripts\activate
python python_cv\detect_yolov8.py
pause
