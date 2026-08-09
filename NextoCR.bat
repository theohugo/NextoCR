@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title NextoCR Manager

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYTHONHASHSEED=42"
set "NEXT_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%NEXT_PYTHON%" (
    echo [NextoCR] Preparation de l'environnement Python ^(premier lancement uniquement^)...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv ".venv"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 goto :python_missing
        python -m venv ".venv"
    )
    if errorlevel 1 goto :setup_failed
)

"%NEXT_PYTHON%" -c "import nextocr_manager, sb3_contrib, stable_baselines3, zmq" >nul 2>nul
if errorlevel 1 (
    echo [NextoCR] Installation des dependances locales d'entrainement...
    "%NEXT_PYTHON%" -m pip install -e "python[train]"
    if errorlevel 1 goto :setup_failed
)

where java >nul 2>nul
if errorlevel 1 if defined JAVA_HOME if exist "%JAVA_HOME%\bin\java.exe" (
    set "PATH=%JAVA_HOME%\bin;%PATH%"
)
where java >nul 2>nul
if errorlevel 1 (
    echo [NextoCR] Java 17 est requis mais n'a pas ete trouve dans PATH.
    echo Installez un JDK 17, puis relancez ce fichier.
    goto :failed
)

echo [NextoCR] Ouverture du manager sur http://127.0.0.1:8765/
"%NEXT_PYTHON%" -m nextocr_manager %*
set "NEXT_EXIT=%ERRORLEVEL%"
if "%NEXT_EXIT%"=="0" exit /b 0

echo.
echo [NextoCR] Le manager s'est arrete avec le code %NEXT_EXIT%.
pause
exit /b %NEXT_EXIT%

:python_missing
echo [NextoCR] Python 3.10 ou plus recent est requis.
echo Installez Python, puis relancez ce fichier.
goto :failed

:setup_failed
echo [NextoCR] La preparation automatique a echoue.
echo Consultez le message ci-dessus, puis relancez ce fichier.

:failed
pause
exit /b 1
