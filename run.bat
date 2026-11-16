@echo off
REM IPA ile verbatim transkripsiyonu calistirir (Windows).
REM "verbatim" adli conda ortamini kullanir; wav\ klasorundeki dosyalari
REM isleyip tg\, txt\, details\ klasorlerine yazar.
REM
REM Konsol penceresi olmadan calistirmak icin (masaustu kisayolu):
REM run_hidden.vbs. O zaman IPA_VERBATIM_HIDDEN=1 olur; buradaki hata
REM mesajlari konsol yerine bir mesaj kutusunda gosterilir ve "pause"
REM yapilmaz (gorunmeyen pencerede sonsuza kadar beklenmesin diye).
REM Not: dosya LF satir sonlu; cmd'nin bu durumdaki etiket (goto/call :x)
REM hatalarina takilmamak icin etiket kullanilmaz.
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Gizli modda hata mesaji kutusu (mesaj IPA_MSG ortam degiskeninden okunur)
set "MSGBOX=powershell -NoProfile -WindowStyle Hidden -Command "Add-Type -AssemblyName PresentationFramework; [void][System.Windows.MessageBox]::Show($env:IPA_MSG, 'ipa4turkish', 'OK', 'Error')""

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
    set "IPA_MSG=HATA: conda bulunamadi. Once verbatim ortamini olusturun: conda env create -f environment.yml"
    if "%IPA_VERBATIM_HIDDEN%"=="1" (
        %MSGBOX%
    ) else (
        echo !IPA_MSG!
        pause
    )
    exit /b 1
)

call "%CONDA_BASE%\condabin\conda.bat" activate verbatim
if errorlevel 1 (
    set "IPA_MSG=HATA: verbatim conda ortami etkinlestirilemedi. Once olusturun: conda env create -f environment.yml"
    if "%IPA_VERBATIM_HIDDEN%"=="1" (
        %MSGBOX%
    ) else (
        echo !IPA_MSG!
        pause
    )
    exit /b 1
)

python "%SCRIPT_DIR%\verbatim.py" %*
set "STATUS=%errorlevel%"

if "%STATUS%"=="0" (
    REM explorer.exe basariyla actiginda bile bazen sifirdan farkli bir
    REM kod dondurebilir; bu yuzden cikis kodu kontrol edilmez.
    start "" explorer "%SCRIPT_DIR%"
) else if "%STATUS%"=="3" (
    REM Kullanici pencerede "Iptal et" / "Durdur" dedi; hata degil.
    exit /b 0
) else if not "%IPA_VERBATIM_HIDDEN%"=="1" (
    REM Gizli modda hata zaten pencerenin gunlugunde ya da ayri bir hata
    REM penceresinde gosterildi; burada beklenmez.
    echo.
    echo Islem hata ile sonlandi ^(cikis kodu %STATUS%^).
    pause
)

exit /b %STATUS%
