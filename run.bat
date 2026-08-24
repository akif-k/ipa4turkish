@echo off
REM IPA ile verbatim transkripsiyonu calistirir (Windows).
REM "verbatim" adli conda ortamini kullanir; wav\ klasorundeki dosyalari
REM isleyip tg\, txt\, details\ klasorlerine yazar.
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "CONDA_BASE="

where conda >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%i in ('conda info --base 2^>nul') do set "CONDA_BASE=%%i"
)

if "%CONDA_BASE%"=="" (
    if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BASE=%USERPROFILE%\miniconda3"
)
if "%CONDA_BASE%"=="" (
    if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BASE=%USERPROFILE%\anaconda3"
)
if "%CONDA_BASE%"=="" (
    if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" set "CONDA_BASE=%LOCALAPPDATA%\miniconda3"
)
if "%CONDA_BASE%"=="" (
    if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_BASE=%ProgramData%\miniconda3"
)

if "%CONDA_BASE%"=="" (
    echo HATA: conda bulunamadi. Once "verbatim" ortamini olusturun:
    echo   conda env create -f environment.yml
    pause
    exit /b 1
)

call "%CONDA_BASE%\condabin\conda.bat" activate verbatim
if errorlevel 1 (
    echo HATA: "verbatim" conda ortami etkinlestirilemedi.
    echo Once olusturun: conda env create -f environment.yml
    pause
    exit /b 1
)

python "%SCRIPT_DIR%\verbatim.py" %*
set "STATUS=%errorlevel%"

if "%STATUS%"=="0" (
    REM explorer.exe basariyla actiginda bile bazen sifirdan farkli bir
    REM kod dondurebilir; bu yuzden cikis kodu kontrol edilmez.
    start "" explorer "%SCRIPT_DIR%"
) else (
    echo.
    echo Islem hata ile sonlandi ^(cikis kodu %STATUS%^).
    pause
)

exit /b %STATUS%
