@echo off
setlocal
python verify_reproducibility.py
if errorlevel 1 exit /b 1
python model_analysis.py
endlocal
