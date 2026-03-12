@echo off
title Claude Code + Swarm Killer
echo.
echo   ============================================
echo     Claude Code + Python Swarm Killer
echo   ============================================
echo.

echo   Currently running:
echo   ------------------------------------------
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE 2>nul | find /i "python"
tasklist /FI "IMAGENAME eq claude.exe" /FO TABLE 2>nul | find /i "claude"
tasklist /FI "IMAGENAME eq node.exe" /FO TABLE 2>nul | find /i "node"
echo   ------------------------------------------
echo.
echo   What do you want to kill?
echo.
echo     1 = Python only  (kills run_swarm)
echo     2 = Claude Code only
echo     3 = Everything (python + claude + node)
echo     Q = Quit
echo.
set /p CHOICE="  Choose (1/2/3/Q): "

if /i "%CHOICE%"=="Q" goto :done
if "%CHOICE%"=="1" goto :kill_python
if "%CHOICE%"=="2" goto :kill_claude
if "%CHOICE%"=="3" goto :kill_all
echo   Invalid choice.
goto :done

:kill_python
echo.
echo   Killing Python processes...
taskkill /F /IM "python.exe" >nul 2>&1
taskkill /F /IM "python3.exe" >nul 2>&1
goto :verify

:kill_claude
echo.
echo   Killing Claude Code processes...
taskkill /F /IM "claude.exe" >nul 2>&1
goto :verify

:kill_all
echo.
echo   Killing everything...
taskkill /F /IM "python.exe" >nul 2>&1
taskkill /F /IM "python3.exe" >nul 2>&1
taskkill /F /IM "claude.exe" >nul 2>&1
taskkill /F /IM "node.exe" >nul 2>&1
goto :verify

:verify
echo.
timeout /t 2 /nobreak >nul
echo   Verification:
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /i "python" >nul
if errorlevel 1 (echo     [OK] No Python processes) else (echo     [!!] Python still running)
tasklist /FI "IMAGENAME eq claude.exe" 2>nul | find /i "claude" >nul
if errorlevel 1 (echo     [OK] No Claude processes) else (echo     [!!] Claude still running)
if "%CHOICE%"=="3" (
    tasklist /FI "IMAGENAME eq node.exe" 2>nul | find /i "node" >nul
    if errorlevel 1 (echo     [OK] No Node processes) else (echo     [!!] Node still running)
)

:done
echo.
pause
