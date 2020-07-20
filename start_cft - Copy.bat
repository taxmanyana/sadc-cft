@echo off

echo.
echo SADC CFT Startup Script
echo.
SET mypath=%~dp0
set mypath=%mypath:~0,-1%

set /p QGIS=<"%mypath%"\qgis.ini
IF %errorlevel% NEQ 0 (
  echo could not locate qgis.ini file, please run the installation script again
  pause
  exit
)
call %QGIS%\bin\o4w_env.bat
call %QGIS%\bin\qt5_env.bat
call %QGIS%\bin\py3_env.bat
echo.
SET PATH=%PATH%;%QGIS%\bin
SET PYTHONPATH=%QGIS%\apps\Python37;%PYTHONPATH%
echo.
python cit.py