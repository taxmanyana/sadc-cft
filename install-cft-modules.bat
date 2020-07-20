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
echo upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% equ 0 (
 echo pip upgraded successfully
)
echo.
python -c "import netCDF4" >nul 2>&1
IF %errorlevel% NEQ 0 (
  echo installing netCDF4...
  python -m pip install --upgrade netCDF4
  if %errorlevel% equ 0 (
    echo netCDF4 installed successfully
  )
) else (
  echo netCDF4 already installed
)
echo.
python -c "import pandas" >nul 2>&1
IF %errorlevel% NEQ 0 (
  echo installing pandas...
  python -m pip install --upgrade pandas
  if %errorlevel% equ 0 (
    echo pandas installed successfully
  )
) else (
  echo pandas already installed
)
echo.
python -c "import setuptools" >nul 2>&1
IF %errorlevel% NEQ 0 (
  echo installing setuptools...
  python -m pip install --upgrade setuptools
  if %errorlevel% equ 0 (
   echo setuptools installed successfully
  )
) else (
  echo setuptools already installed
)
echo.
python -c "import sklearn" >nul 2>&1
IF %errorlevel% NEQ 0 (
  echo installing sklearn...
  python -m pip install --upgrade sklearn
  if %errorlevel% equ 0 (
   echo sklearn installed successfully
  )
) else (
  echo sklearn already installed
)
echo.
python -c "import statsmodels" >nul 2>&1
IF %errorlevel% NEQ 0 (
  echo installing statsmodels...
  python -m pip install --upgrade statsmodels
  if %errorlevel% equ 0 (
   echo statsmodels installed successfully
  )
) else (
  echo statsmodels already installed
)
echo.
python -c "import scipy" >nul 2>&1
IF %errorlevel% NEQ 0 (
echo installing scipy...
python -m pip install --upgrade scipy
if %errorlevel% equ 0 (
 echo scipy installed successfully
)
) else (
  echo scipy already installed
)
echo.
python -c "import geojson" >nul 2>&1
IF %errorlevel% NEQ 0 (
  echo installing geojson...
  python -m pip install --upgrade geojson
  if %errorlevel% equ 0 (
   echo geojson installed successfully
  )
) else (
  echo geojson already installed
)
echo.
python -c "import numpy" >nul 2>&1
IF %errorlevel% NEQ 0 (
  echo installing numpy...
  python -m pip install --upgrade numpy
  if %errorlevel% equ 0 (
   echo numpy installed successfully
  )
) else (
  echo numpy already installed
)
echo.
python -c "import shapely" >nul 2>&1
IF %errorlevel% NEQ 0 (
  echo installing shapely...
  python -m pip install --upgrade shapely
  if %errorlevel% equ 0 (
   echo shapely installed successfully
  )
) else (
  echo shapely already installed
)
echo.
pause
exit

