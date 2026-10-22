@echo off
REM Lokal backend: model shu kompyuterda, sahifa Vercelda. Toxtatish: Ctrl+C
cd /d "%~dp0..\.."
python deploy\local\start_backend.py
pause
