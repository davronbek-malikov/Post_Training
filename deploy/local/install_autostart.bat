@echo off
rem Kompyuter yonib, Windows'ga kirganingizda TeachHelp server o'zi ishga tushsin va zaryadlovchiga ulangan
rem holatda kompyuter uxlab qolmasin. Administrator huquqi kerak emas: yorliq foydalanuvchining Startup
rem papkasiga qo'yiladi (Task Scheduler'ning ONLOGON vazifasi esa admin talab qiladi). Bekor qilish: uninstall_autostart.bat
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Startup') + '\TeachHelp Server.lnk'); $s.TargetPath = (Get-Command pythonw.exe).Source; $s.Arguments = '\"%~dp0run.py\"'; $s.WorkingDirectory = '%~dp0'; $s.Save()"
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
echo.
echo Tayyor: Windows'ga kirganingizda server o'zi ishga tushadi; zaryadlovchida uxlash o'chirildi.
echo Eslatma: kompyuter qayta yonganda Windows'ga kirish (login) kerak - server shundan keyin ishlaydi.
