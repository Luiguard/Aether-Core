@echo off
title Aether-Core Launcher
color 0B
echo ==============================================================
echo                 AETHER-CORE STARTUP DASHBOARD
echo ==============================================================
echo.
echo Initialisiere Umgebung...
echo.

:: Prüfe ob Python installiert ist
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [Fehler] Python wurde nicht gefunden! Bitte installiere Python.
    pause
    exit /b
)

:: Führe den Launcher aus
python aether_launcher.py

pause
