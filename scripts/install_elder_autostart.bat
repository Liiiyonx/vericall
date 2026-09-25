@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在把「适老一体机管家」加入开机自启（开始菜单-启动文件夹）...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root=(Resolve-Path '%~dp0..').Path; " ^
  "$lnk=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path ([Environment]::GetFolderPath('Startup')) 'VeriCall-Elder.lnk')); " ^
  "$lnk.TargetPath=Join-Path $root 'start_elder.bat'; " ^
  "$lnk.WorkingDirectory=$root; " ^
  "$lnk.Description='VeriCall Elder Kiosk (autostart)'; " ^
  "$lnk.Save(); Write-Output ('Installed: '+$lnk.FullName)"
echo.
echo 完成。取消自启：删除启动文件夹里的 VeriCall-Elder.lnk 即可。
pause