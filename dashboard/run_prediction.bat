@echo off
setlocal

:: Get image path argument
set "IMG=%~1"

:: Define paths
set "ROOT=%~dp0.."
set "PYTHON=%ROOT%\venv\Scripts\python.exe"
set "MAIN=%ROOT%\main.py"
set "OUTPUT=%ROOT%\results\output.txt"

echo Running prediction on %IMG% > "%OUTPUT%"

:: Run main.py and capture output
"%PYTHON%" "%MAIN%" -i "%IMG%" -m both >> "%OUTPUT%" 2>&1

echo DONE >> "%OUTPUT%"
endlocal
