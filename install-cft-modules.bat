@echo off

echo.
echo SADC CFT Installation Script
echo.
SET mypath=%~dp0
set mypath=%mypath:~0,-1%

set QGIS=""
rem Detect QGIS Version
FOR /F "tokens=* USEBACKQ" %%g in (`dir /b "C:\Program Files (x86)\QGIS 3*" 2^>NUL`) do (
  set QGIS="C:\Program Files (x86)\%%g"
)
FOR /F "tokens=* USEBACKQ" %%g in (`dir /b "C:\Program Files\QGIS 3*" 2^>NUL`) do (
  set QGIS="C:\Program Files\%%g"
)

if %QGIS% == "" (
  echo.
  echo could not find any QGIS v3 installation
  pause
  exit
)
echo %QGIS%> "%mypath%"\qgis.ini
call %QGIS%\bin\o4w_env.bat
call %QGIS%\bin\qt5_env.bat
call %QGIS%\bin\py3_env.bat
echo.

echo QGIS installation to be used: %QGIS%
echo.
echo installing the python modules...
echo.
echo upgrading numpy...
python -m pip install --upgrade numpy
if %errorlevel% equ 0 (
 echo numpy upgraded successfully
)
echo.
echo upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% equ 0 (
 echo pip upgraded successfully
)
echo.
echo upgrading netCDF4...
python -m pip install --upgrade netCDF4
if %errorlevel% equ 0 (
 echo netCDF4 upgraded successfully
)
echo.
echo upgrading pandas...
python -m pip install --upgrade pandas
if %errorlevel% equ 0 (
 echo pandas upgraded successfully
)
echo.
echo upgrading setuptools...
python -m pip install --upgrade setuptools
if %errorlevel% equ 0 (
 echo setuptools upgraded successfully
)
echo.
echo upgrading sklearn...
python -m pip install --upgrade sklearn
if %errorlevel% equ 0 (
 echo sklearn upgraded successfully
)
echo.
echo upgrading statsmodels...
python -m pip install --upgrade statsmodels
if %errorlevel% equ 0 (
 echo statsmodels upgraded successfully
)
echo.
echo upgrading scipy...
python -m pip install --upgrade scipy
if %errorlevel% equ 0 (
 echo scipy upgraded successfully
)
echo.
echo upgrading geojson...
python -m pip install --upgrade geojson
if %errorlevel% equ 0 (
 echo geojson upgraded successfully
)
echo.
echo upgrading shapely...
python -m pip install --upgrade shapely
if %errorlevel% equ 0 (
 echo shapely upgraded successfully
)
echo.
echo upgrading descartes...
python -m pip install --upgrade descartes
if %errorlevel% equ 0 (
 echo descartes upgraded successfully
)
echo.
pause
exit

