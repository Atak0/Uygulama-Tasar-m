@echo off
cd /d "%~dp0"

echo ==========================================================
echo   SpineAI - Omurga X-Ray Analiz Sistemi
echo ==========================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [!] .venv bulunamadi.
    echo.
    echo Once su komutlari calistirin:
    echo   py -3.9 -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo [+] Flask sunucusu baslatiliyor...
echo [+] Adres: http://localhost:5000
echo.

python webapp\app.py
pause
