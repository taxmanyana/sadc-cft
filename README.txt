SADC Climate Forecasting Tool (SADC CFT)
SADC Climate Services Centre
SARCIS-DR Project


CREDITS
-------
Programmer: Thembani Moitlhobogi
Theory and Formulas: Sunshine Gamedze, Dr Arlindo Meque, Climate Experts from SADC NHMSs
Initial version release date: 23 August 2019
Subsequent releases detailed in the "sadc_cft_versionlog.txt" file
Funding: African Development Bank, European Union, SADC


INTRODUCTION
------------
The SADC Climate Services Centre (CSC) is developing the SADC Climate Forecasting Tool (SADC CFT), 
which is a statistical seasonal forecasting tool. It uses Quantum GIS (QGIS) version 3 core utilities, 
and is programmed in the Python language.

The CFT is being developed by the CSC under the Southern African Regional Climate Information Services 
for Disaster Resilience Development (SARCIS-DR) Project. The main objective of the tool is to support 
CSC and NMHSs to automate climate data products generation. 


INSTALLATION
------------
1. Download and Install the latest version of QGIS 3
2. Download the latest version of SADC CFT from: ftp://cscftp.sadc.int/software/cft/ 
3. Unpack (unzip) the sadc_cft-x.x.x.zip ZIP file to a directory of your choosing (e.g. My Documents)
4. Navigate into the extracted folder "sadc_cft-x.x.x" 
5. Right-click the "install-cft-modules.bat" and select "Run as Administrator" to install the required python modules (internet connection required)
6. Once all the python modules have been successfully installed, the CFT ready to run


RUNNING THE SADC CFT
--------------------
SADC CFT is easily run by just double-clicking on "start_cft.bat", the application sometimes opens minimized so keep an eye on the task bar.


FEATURES
--------
- Can easily select a year to forecast
- Option to forecast a 3-month season, or a single month
- Adjustable model training period
- Supports multiple predictors (in NetCDF format). Tool will loop and generate forecast for each predictor
- Supports point (station) as the predictant
- Supports season cumulation (rainfall) or season average (temperature) functions
- Supports Linear Regression and Multilayer Perceptron regression techniques
- A forecast can be summarized per zone, where a weighted average forecast is produced from the stations within the zone
- Computes skill scores for each forecast

Future developments
- Validates forecasts, if ground data is available for forecast period
- Support for gridded ground data (NetCDF) as the predictant






