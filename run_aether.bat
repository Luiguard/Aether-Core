@echo off
setlocal
cls
title Aether-Core Commander

echo ==============================================================
echo.
echo           A E T H E R - C O R E   U N I F I E D
echo.
echo ==============================================================
echo.
echo Dieses Programm startet das gesamte Aether-System.
echo.

echo [1/2] Starte Hintergrund-Services (API, Web, Agent)...
:: Wir starten den Launcher in einem separaten Hintergrund-Fenster,
:: damit dieses CMD-Fenster fuer die Training-Option frei bleibt.
start /b python aether_launcher.py

timeout /t 5 /nobreak > nul

echo.
set /p DO_TRAIN="Moechtest du ZUSAETZLICH 60 Epochen initial trainieren? (y/n) [Default: n]: "

if /i "%DO_TRAIN%"=="y" (
    echo.
    echo Starte initiales Trainings-Paket... 
    python distill.py 60
)

echo.
echo ==============================================================
echo Alle Systeme sind aktiv. Viel Spaß mit Aether-Core!
echo ==============================================================
echo.
echo Dieses Fenster kann offen bleiben, um Logs zu sehen.
pause
