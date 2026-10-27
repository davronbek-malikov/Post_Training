@echo off
rem install_autostart.bat ni bekor qilish: avtomatik ishga tushish o'chiriladi, uxlash rejimi 30 daqiqaga qaytariladi.
schtasks /Delete /F /TN "TeachHelp Server"
powercfg /change standby-timeout-ac 30
echo Avtomatik ishga tushish o'chirildi.
