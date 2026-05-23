@echo off
title Arogya Healthcare Assistant Launcher
echo ============================================================
echo      Arogya Healthcare Assistant Premium Launcher
echo ============================================================
echo.

:: Check if virtual environment exists
if not exist "chatbot_env" (
    echo [ERROR] Virtual environment 'chatbot_env' not found in this folder.
    echo Please ensure the 'chatbot_env' folder is present in the current directory.
    pause
    exit /b
)

echo [SYSTEM] Activating Python virtual environment (chatbot_env)...
call chatbot_env\Scripts\activate.bat

echo [SYSTEM] Checking required libraries...
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Streamlit is not detected in 'chatbot_env'.
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
)

:: Launcher loaded successfully

echo.
echo [SYSTEM] Starting Streamlit Server...
echo ============================================================
echo.
streamlit run app.py

pause
