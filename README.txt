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
The CFT is being developed by the CSC under the Southern African Regional Climate Information Services for Disaster Resilience Development (SARCIS-DR) Project. The main objective of the tool is to support the CSC and NMHSs to automate the manual statistical process for generating a seasonal forecast outlook for the Southern African Regional Climate Outlook Forum. The main programming language used is Python version 3.


LICENSE
-------
The CFT code and software is licensed under the Zero-Clause BSD license.
Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.

Copyright (C) 2020 by SADC-CSC <csc1@sadc.int>


DOWNLOADING CFT
------------
The latest version of CFT is available from the SADC-CSC FTP site:  ftp://cscftp.sadc.int/software/cft/ 


INSTALLATION ON WINDOWS
--------------------
1. Ensure you have installed the latest version of QGIS 3 in your computer
2. Unpack (unzip) the sadc_cft-x.x.x.zip ZIP file to a directory of your choosing (e.g. My Documents)
3. Navigate into the extracted folder "sadc_cft-x.x.x" 
4. Right-click the "install-cft-modules.bat" and select "Run as Administrator" to install the required python modules (internet connection required)
5. Once installed, CFT is easily be run by double-clicking on "start_cft.bat"


INSTALLATION ON LINUX
--------------------
The installation script will download dependency sources, compile and deploy the CFT
1. Unpack (unzip) the sadc_cft-x.x.x.zip ZIP file to a directory of your choosing
2. On the terminal, navigate into the extracted folder:  cd sadc_cft-x.x.x 
3. Run the installation script using the following command:   ./install-cft-linux.sh
4. Once installed, you can run CFT from the terminal using the following commands:
   source python3/bin/activate
   python3 cft.py settings.json
5. CFT also has a Desktop launcher "CFT.desktop" which will be installed to the Desktop. On the Desktop double-click on "CFT.desktop" to run the tool (if it is the first time then it will bring a pop-up for you to trust and accept)
6. The MPI version can be run on the terminal by executing the following commands:
   source python3/bin/activate
   mpirun -n 40 python3 cft_mpi.py settings.json


ALTERNATIVE INSTALLATION ON LINUX (FOR UBUNTU ONLY)
--------------------
1. On the terminal, install the required dependencies using the command:   sudo apt install python3-pip python3-venv libffi-dev
2. Unpack (unzip) the sadc_cft-x.x.x.zip ZIP file to a directory of your choosing
3. On the terminal, navigate into the extracted folder:  cd sadc_cft-x.x.x
4. Run the installation script using the following command:   ./install-cft-ubuntu.sh
5. Once installed, you can run CFT from the terminal using command:    ./cft_ubuntu.sh
6. CFT also has a Desktop launcher "CFT.desktop" which will be copied to the Desktop. On the Desktop double-click on "CFT.desktop" to run the tool (if it is the first time then it will bring a pop-up for you to trust and accept)


FEATURES
--------
- Can easily select a year to forecast
- Option to forecast a 3-month season, or a single month
- Adjustable model training period
- Supports multiple predictors (in NetCDF format). Tool will loop and generate forecast for each predictor
- Support for gridded ground data (NetCDF) as the predictant
- Supports point (station) as the predictant
- Supports season cumulation (rainfall) or season average (temperature) functions
- Supports Linear Regression and Artificial Intelligence (Multilayer Perceptron regression)
- For station (point) data, a forecast can be summarized per zone, where a weighted average forecast is produced from the stations within the zone
- Computes skill scores for each forecast
