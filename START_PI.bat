@echo off
title ScareX - Standalone Raspberry Pi 5 System
echo ==========================================================
echo        Starting ScareX Standalone System (Pi 5 Mode)
echo ==========================================================
cd /d "%~dp0"
python -m raspberry_pi.main
pause
