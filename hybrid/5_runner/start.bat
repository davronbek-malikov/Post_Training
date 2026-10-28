@echo off
rem TeachHelp lokal serverini oynasiz (fon rejimida) ishga tushirish. Havola: deploy\local\data\backend_link.txt
rem Birinchi marta oldin:  python deploy\local\run.py --admin siz@gmail.com   (sozlamalarni yozadi, Ctrl+C bilan to'xtating)
start "" pythonw "%~dp0run.py"
echo TeachHelp server fon rejimida ishga tushdi. 1-2 daqiqadan keyin havola: %~dp0data\backend_link.txt
echo Log: %~dp0data\run.log    To'xtatish: %~dp0stop.bat
