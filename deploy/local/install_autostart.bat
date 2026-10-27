@echo off
rem Kompyuter yonib, Windows'ga kirganingizda TeachHelp server o'zi ishga tushsin (Task Scheduler) va
rem zaryadlovchiga ulangan holatda kompyuter uxlab qolmasin. Bekor qilish: uninstall_autostart.bat
schtasks /Create /F /SC ONLOGON /DELAY 0001:00 /TN "TeachHelp Server" /TR "pythonw.exe \"%~dp0run.py\""
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
echo.
echo Tayyor: Windows'ga kirgandan 1 daqiqa keyin server o'zi ishga tushadi; zaryadlovchida uxlash o'chirildi.
echo Eslatma: kompyuter qayta yonganda Windows'ga kirish (login) kerak - vazifa shundan keyin ishlaydi.
