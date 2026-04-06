@echo off
setlocal
cls
title Aether-Core Knowledge Distillation

echo ==============================================================
echo.
echo           A E T H E R - C O R E   -   T R A I N E R
echo.
echo ==============================================================
echo.
echo Dieses Programm trainiert das lokale neuronale Modell.
echo Wissen wird vom DeepSeek-Teacher "destilliert".
echo.

:: Die API-Key Abfrage ist nun im UI moeglich, 
:: aber fuer manuelle Runs hier als Noob-Sicherheit.
if "%AETHER_TEACHER_API_KEY%"=="" (
    echo [HINWEIS] Es ist kein API-Key in der Sitzung gefunden worden. 
    echo Ohne KEY nutzt Aether nur den Dummy-Modus!
    echo (Du kannst den Key auch permanent im Browser-Dashboard speichern!)
    echo.
)

set /p EPOCHS="Wie viele Epochen moechtest du trainieren? (Default: 60): "
if "%EPOCHS%"=="" set EPOCHS=60

echo.
echo Starte Destillation fuer %EPOCHS% Epochen... 
echo.
python distill.py %EPOCHS%

echo.
echo Training abgeschlossen. Du kannst nun das Dashboard wieder oeffnen!
pause
