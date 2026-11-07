@echo off
cd /d "%~dp0"
set "VIDEO_PATH="
set /p "VIDEO_PATH=Enter an input video path: "
if not defined VIDEO_PATH (
	echo A video path is required.
	pause
	exit /b 1
)
call python_cv\venv\Scripts\activate
python python_cv\main.py "%VIDEO_PATH%"
pause
