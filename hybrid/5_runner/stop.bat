@echo off
rem TeachHelp lokal serverini to'xtatish: run.py, llama-server, gateway (uvicorn) va cloudflared tunnel.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*hybrid\5_runner\run.py*' -or $_.CommandLine -like '*uvicorn gateway:app*' -or $_.Name -in 'llama-server.exe','cloudflared.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host 'to''xtatildi:' $_.Name $_.ProcessId }"
