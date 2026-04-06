@echo off
title Aether-Core Distillation Trainer
color 0D
echo ==============================================================
echo                 AETHER-CORE KNOWLEDGE DISTILLATION
echo ==============================================================
echo.
echo Dieses Programm trainiert das lokale neuronale Modell,
echo indem es Wissen (Sprachmuster und Fakten) vom DeepSeek-Teacher "destilliert".
echo.

:: Prüfe ob API_KEY vorhanden ist, falls nicht, frage danach
if "%AETHER_TEACHER_API_KEY%"=="" (
    echo [HINWEIS] Es ist kein API-Key gefunden worden. 
    echo Ohne DeepSeek-R2 nutzt Aether nur Dummy-Saetze zum Training!
    set /p AETHER_TEACHER_API_KEY="Bitte DeepSeek API-Key eingeben (oder Enter fuer Offline-Modus): "
)

echo.
set /p EPOCHS="Wie viele Epochen moechtest du trainieren? (Zahl eingeben, z.B. 60 oder 100): "
if "%EPOCHS%"=="" set EPOCHS=60

echo.
echo ==============================================================
echo Starte Destillation fuer %EPOCHS% Epochen... 
echo ==============================================================
python distill.py --config config.yaml --epochs %EPOCHS%

echo.
echo Training abgeschlossen. Du kannst nun das Dashboard wieder oeffnen!
pause
