# SADC Climate Services Centre
# SARCIS-DR Project
# Programming: Thembani Moitlhobogi
# Theory and Formulas: Sunshine Gamedze, Dr Arlindo Meque, Climate Experts from SADC NHMSs
# July 2020
#
import os, sys, re, time
from dateutil.relativedelta import relativedelta
from datetime import date, datetime
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import geojson, json
import threading
from multiprocessing import Pool
from functools import partial
from functions import *

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QThread, QObject, QDate, QTime, QDateTime, Qt
qtCreatorFile = "CIT.ui"

# Global Variables
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
seasons = ['JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ','DJF']
csvheader = 'Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec'
fcstyear = QDate.currentDate().year()
settingsfile = 'settings.json'
predictordict = {}
predictantdict = {}
predictantdict['stations'] = []
predictantdict['data'] = None
fcstPeriod = None

if os.path.isfile(settingsfile):
    with open(settingsfile, "r") as read_file:
        config = json.load(read_file)
else:
    config = {}
    config['outDir'] = ''
    config['predictorList'] = []
    config['predictantList'] = []
    config['predictantMissingValue'] = -9999
    config['fcstPeriodType'] = 0
    config['fcstPeriodIndex'] = 9
    config['trainStartYear'] = 1971
    config['trainEndYear'] = 2000
    config['predictorMonthIndex'] = 5
    config['enableLR'] = True
    config['PValue'] = 0.05
    config['selectMode'] = 1
    config['stepwisePvalue'] = 0.3
    config['neurons'] = "25,25"
    config['inputFormat'] = "CSV"
    config['includeScore'] = False
    config['composition'] = "Cumulation"
    config['zonevector'] = {"file": None, "ID": 0, "attr": []}
    #config['zoneID'] = None
    config['fcstyear'] = fcstyear
    config['algorithms'] = []
    config['basinbounds'] = {"minlat": -90, "maxlat": 90, "minlon": -180, "maxlon": 180}

#
def concat_csvs(csvs, missing):
    dfs_files = []
    for file in csvs:
        dfs_files.append(pd.read_csv(file, encoding = 'ISO-8859-9'))
    dfs_files = pd.concat((dfs_files), axis=0)
    dfs_files = dfs_files.replace(missing, np.nan)
    dfs_files = dfs_files.dropna(how='all')
    return dfs_files

def get_parameter(list):
    keys = []
    for key in list:
        keys.append(key)
    ref_keys = ['Y', 'X', 'T','zlev']
    for x in reversed(range(len(keys))):
        if keys[x] not in ref_keys:
            return keys[x]
    return None

#
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()

    def getOutDir():
        global config
        config['outDir'] = QtWidgets.QFileDialog.getExistingDirectory()
        window.outdirlabel.setText(config.get('outDir'))

    def addPredictors():
        global config
        fileNames = QtWidgets.QFileDialog.getOpenFileNames(window,
                    'Add File(s)', './', filter="NetCDF File (*.nc*)")
        for fileName in fileNames[0]:
            config['predictorList'].append(fileName)
            window.predictorlistWidget.addItem(os.path.basename(fileName))

    def removePredictors():
        global config
        newList = []
        for yy in config.get('predictorList'):
            if os.path.basename(yy) != window.predictorlistWidget.selectedItems()[0].text():
                newList.append(yy)
        window.predictorlistWidget.clear()
        config['predictorList'] = newList
        for yy in newList:
            window.predictorlistWidget.addItem(os.path.basename(yy))

    def addPredictants():
        global config
        global csvheader
        config['predictantList'] = []
        window.predictantlistWidget.clear()
        if config['inputFormat'] == "CSV":
            fileNames = QtWidgets.QFileDialog.getOpenFileNames(window,
                    'Add File(s)', './', filter="CSV File (*.csv)")
            for filename in fileNames[0]:
                with open(filename) as f:
                    fline = f.readline().rstrip()
                if fline.count(',') < 12:
                    window.progresslabel.setText(
                        "Format error in "+os.path.basename(filename)+", check if comma delimited")
                    continue
                if csvheader not in fline:
                    window.progresslabel.setText(
                        "Format error, one or more column headers incorrect in " + os.path.basename(filename))
                    continue
                config['predictantList'].append(filename)
                window.predictantlistWidget.addItem(os.path.basename(filename))
        else:
            fileName = QtWidgets.QFileDialog.getOpenFileName(window,
                    'Add File', './', filter="NetCDF File (*.nc*)")
            config['predictantList'].append(fileName[0])
            window.predictantlistWidget.addItem(os.path.basename(fileName[0]))

    def clearPredictants():
        global config
        config['predictantList'] = []
        window.predictantlistWidget.clear()

    def change_period_list():
        global config
        periodlist = []
        window.periodComboBox.clear()
        if window.period1Radio.isChecked() == True:
            config['fcstPeriodType'] = 0
            periodlist = seasons
        if window.period2Radio.isChecked() == True:
            config['fcstPeriodType'] = 1
            periodlist = months
        for xx in range(len(periodlist)):
            window.periodComboBox.addItem(periodlist[xx])

    def populate_period_list(period, index):
        periodlist = []
        window.periodComboBox.clear()
        if period == 0:
            window.period1Radio.setChecked(True)
            periodlist = seasons
        if period == 1:
            window.period2Radio.setChecked(True)
            periodlist = months
        for xx in range(len(periodlist)):
            window.periodComboBox.addItem(periodlist[xx])
        window.periodComboBox.setCurrentIndex(index)


    def addZoneVector():
        global config
        window.zoneIDcomboBox.clear()
        window.zonevectorlabel.setText('')
        config['zonevector'] = {"file": None, "ID": 0, "attr": []}
        zonefieldsx = []
        window.zoneIDcomboBox.setDuplicatesEnabled(False)
        fileName = QtWidgets.QFileDialog.getOpenFileName(window,
                  'Add File', './', filter="GeoJson File (*.geojson)")
        config['zonevector']['file'] = fileName[0]
        if os.path.isfile(config.get('zonevector',{}).get('file')):
            with open(config.get('zonevector',{}).get('file')) as f:
                zonejson = geojson.load(f)
            for zonekey in zonejson['features']:
                for zonetype in zonekey.properties:
                    zonefieldsx.append(zonetype)
            zonefields = []
            [zonefields.append(x) for x in zonefieldsx if x not in zonefields]
            for xx in zonefields:
                window.zoneIDcomboBox.addItem(str(xx))
                config['zonevector']['attr'].append(str(xx))
            window.zonevectorlabel.setText(os.path.basename(config.get('zonevector',{}).get('file')))

    def setInputFormat():
        global config
        if window.CSVRadio.isChecked():
            config['inputFormat'] = "CSV"
        else:
            config['inputFormat'] = "NetCDF"

    def setInclude():
        global config
        if window.inclscorecheckBox.isChecked():
            config['includeScore'] = True
        else:
            config['includeScore'] = False

    for xx in range(len(months)):
        window.predictMonthComboBox.addItem(months[xx])

    def launch_forecast_Thread():
        t = threading.Thread(target=forecast)
        t.start()

    def forecast():
        global settingsfile
        global config
        global predictordict
        global predictantdict
        global fcstPeriod
        window.progresslabel.setText('preparing inputs')
        config['algorithms'] = []
        if window.LRcheckBox.isChecked():
            config['algorithms'].append('LR')
        # if window.NNcheckBox.isChecked():
        if window.LSTMcheckBox.isChecked(): config['algorithms'].append('LSTM')
        if window.MLPcheckBox.isChecked(): config['algorithms'].append('MLP')
        if window.CNNcheckBox.isChecked(): config['algorithms'].append('CNN')
        config['neurons'] = window.neuronslineEdit.text()

        if window.cumRadio.isChecked():
            config['composition'] = "Cumulation"
        else:
            config['composition'] = "Average"

        if window.period1Radio.isChecked():
            config['fcstPeriodType'] = 0
            config['fcstPeriodIndex'] = seasons.index(str(window.periodComboBox.currentText()))
        else:
            config['fcstPeriodType'] = 1
            config['fcstPeriodIndex'] = months.index(str(window.periodComboBox.currentText()))
        config['predictorMonthIndex'] = window.predictMonthComboBox.currentIndex()
        config['stepwisePvalue'] = float(window.swpvaluelineEdit.text())
        config['PValue'] = float(window.pvaluelineEdit.text())
        config['fcstyear'] = int(window.fcstyearlineEdit.text())
        config['zonevector']['ID'] = window.zoneIDcomboBox.currentIndex()
        config['basinbounds']['minlat'] = window.minlatLineEdit.text()
        config['basinbounds']['maxlat'] = window.maxlatLineEdit.text()
        config['basinbounds']['minlon'] = window.minlonLineEdit.text()
        config['basinbounds']['maxlon'] = window.maxlonLineEdit.text()

        # check if output directory exists
        if not os.path.exists(config.get('outDir')):
            window.progresslabel.setText("Output Directory not set!")
            return

        processes = len(config.get('predictorList')) * len(config.get('algorithms'))

        # Write configuration to settings file
        import json
        with open(settingsfile, 'w') as fp:
            json.dump(config, fp, indent=4)

        # prepare input data
        nstations = 0
        if config.get('inputFormat') == 'CSV':
            if len(config.get('predictantList')) != 0:
                missing = config.get('predictantMissingValue')
                if len(str(missing)) == 0: missing = -9999
                input_data = concat_csvs(config.get('predictantList'), missing)
                predictantdict['data'] = input_data
                stations = list(input_data['ID'].unique())
                predictantdict['stations'] = stations
                nstations = len(stations)
                lat, lon = [], []
                for n in range(nstations):
                    station_data_all = input_data.loc[input_data['ID'] == stations[n]]
                    lat.append(station_data_all['Lat'].unique()[0])
                    lon.append(station_data_all['Lon'].unique()[0])
                fcst_precip = np.zeros(nstations)
                fcst_class = np.zeros(nstations)
                qt1 = np.zeros(nstations)
                qt2 = np.zeros(nstations)
                qt3 = np.zeros(nstations)
                prmean = np.zeros(nstations)
                skills = np.zeros(nstations)
                probabilities = [None] * nstations
            else:
                input_data = None

        predictorEndYr = int(config.get('fcstyear'))
        predictorStartYr = int(config.get('trainStartYear'))
        predictorMonth = str(window.predictMonthComboBox.currentText())
        fcstPeriod = str(window.periodComboBox.currentText())

        cproc = 0

        for predictor in config.get('predictorList'):
            if os.path.isfile(predictor):
                predictorName = os.path.splitext(os.path.basename(predictor))[0]
                predictordict[predictorName] = {}
                window.progresslabel.setText('processing ' + predictorName)
                print('checking ' + predictorName)
                dataset = Dataset(predictor)
                sstmon = month_dict.get(predictorMonth.lower(), None)
                ref_date = str(dataset.variables['T'].units).split("since ", 1)[1]
                ref_date = datetime.strptime(ref_date, '%Y-%m-%d')
                param = get_parameter(dataset.variables.keys())
                timearr = np.array(dataset.variables['T'][:], dtype=int)
                sst = dataset.variables[param][:]
                predictordict[predictorName]['lats'] = dataset.variables['Y'][:]
                predictordict[predictorName]['lons'] = dataset.variables['X'][:]
                rows = len(predictordict[predictorName]['lats'])
                cols = len(predictordict[predictorName]['lons'])
                mon_arr = []
                year_arr = []
                sst_index = []
                x = 0
                for mon in timearr:
                    month = (ref_date + relativedelta(months=+mon)).strftime("%m")
                    if month == sstmon:
                        mon_arr.append(mon)
                        sst_index.append(x)
                        x = x + 1

                if len(mon_arr) == 0:
                    status = "Predictor ("+param+") does not contain any data for " + predictorMonth
                    window.progresslabel.setText(status)
                    continue

                for mon in mon_arr:
                    year_arr.append(int((ref_date + relativedelta(months=+int(mon))).strftime("%Y")))

                if config.get('predictorMonthIndex') >= config.get('fcstPeriodIndex'):
                    predictorStartYr = int(config.get('trainStartYear')) - 1
                    predictorEndYr = int(config.get('fcstyear')) - 1

                if int(config.get('fcstyear')) > max(year_arr):
                    predictorStartYr = config.get('trainStartYear') - 1
                    predictorEndYr = int(config.get('fcstyear')) - 1
                    if  int(config.get('fcstyear')) - max(year_arr) > 1:
                        status = "Predictor ("+param+") for " + predictorMonth + " goes up to " + str(year_arr[-1]) + \
                            ", cannot be used to forecast " + str(config.get('fcstyear')) + ' ' + fcstPeriod
                        window.progresslabel.setText(status)
                        continue
                    if config.get('fcstPeriodIndex') >= config.get('predictorMonthIndex'):
                        status = "Predictor ("+param+") for " + predictorMonth + " goes up to " + str(year_arr[-1]) + \
                            ", cannot be used to forecast " + str(config.get('fcstyear')) + ' ' + fcstPeriod
                        window.progresslabel.setText(status)
                        continue

                if int(config.get('fcstyear')) <= config.get('trainEndYear'):
                    status = "Cannot forecast " + str(config.get('fcstyear')) + " as it is not beyond training period"
                    window.progresslabel.setText(status)
                    continue

                if predictorStartYr < year_arr[0]:
                    status = "Predictor ("+param+") data starts in " + str(year_arr[0]) + \
                        ", selected options require predictor to start in " + str(predictorStartYr)
                    window.progresslabel.setText(status)
                    continue

                status = 'predictor data to be used: ' + str(predictorStartYr) + predictorMonth + ' to ' + \
                         str(predictorEndYr) + predictorMonth
                window.progresslabel.setText(status)
                # yearspredictor = [yr for yr in range(predictorStartYr, predictorEndYr+1)]
                nsst_years = predictorEndYr - predictorStartYr + 1
                sst_arr = np.zeros((nsst_years, rows, cols)) * np.nan
                x = -1
                for year in range(predictorStartYr, predictorEndYr + 1):
                    x = x + 1
                    try:
                        # get index of year from list
                        i = year_arr.index(year)
                        j = mon_arr[i]
                        k = list(timearr).index(j)
                        sst_arr[x] = np.array(sst[k])
                    except:
                        continue

                predictordict[predictorName]['ref_date'] = ref_date
                predictordict[predictorName]['param'] = param
                predictordict[predictorName]['predictorMonth'] = predictorMonth
                predictordict[predictorName]['time'] = timearr
                predictordict[predictorName]['year_arr'] = year_arr
                predictordict[predictorName]['data'] = sst_arr
                predictordict[predictorName]['predictorStartYr'] = predictorStartYr
                predictordict[predictorName]['predictorEndYr'] = predictorEndYr
                sst_arr = None
                sst = None

        # print(forecast_points(config, predictordict, predictantdict, fcstPeriod, stations[0]))
        func = partial(forecast_points, config, predictordict, predictantdict, fcstPeriod)

        p = Pool(6)
        rs = p.imap_unordered(func, stations)
        p.close()
        while (True):
            completed = rs._index
            if (completed >= len(stations)): break
            status = "completed processing of " + str(completed) + " of " + str(len(stations)) + " stations"
            window.progresslabel.setText(status)
            # print("processing ", completed, " of ", len(stations))
            time.sleep(0.2)
        r = list(rs)
        print(r)

        #         for algorithm in config.get('algorithms'):
        #             cproc = cproc + 1
        #             outpath = config.get('outDir') + os.sep + 'Forecast_' + str(config.get('fcstyear')) + '_' + fcstPeriod + os.sep + algorithm
        #             cstatus = '['+str(cproc)+'/'+str(processes)+'] '
        #             if algorithm == 'LR':
        #                 if config.get('inputFormat') == 'CSV':
        #                     # if a zone file has been defined...
        #                     if config.get('zonevector',{}).get('file') is not None:
        #                         if os.path.isfile(config.get('zonevector',{}).get('file')):
        #                             with open(config.get('zonevector',{}).get('file')) as f:
        #                                 zonejson = geojson.load(f)
        #                             zoneattrID = config.get('zonevector',{}).get('ID')
        #                             zoneattr = config.get('zonevector', {}).get('attr')[zoneattrID]
        #                             zones = zonelist(zonejson, zoneattr)
        #                             zonestation = {}
        #                             fcstzone = {}
        #                             # --------------
        #                             for n in range(nstations):
        #                                 station_data_all = input_data.loc[input_data['ID'] == stations[n]]
        #                                 y = station_data_all['Lat'].unique()[0]
        #                                 z = station_data_all['Lon'].unique()[0]
        #                                 szone = whichzone(zonejson, y, z, zoneattr)
        #                                 try:
        #                                     zonestation[szone].append(stations[n])
        #                                 except KeyError:
        #                                     zonestation[szone] = [stations[n]]
        #                             window.progresslabel.setText(cstatus + "Calculating forecast for each zone...")
        #                             nzones = len(zones)
        #                             czone = 0
        #                             for zone in zones:
        #                                 czone = czone + 1
        #                                 window.progresslabel.setText(cstatus + 'processing zone ' + str(czone) +
        #                                            ' of ' + str(nzones) + ' (' + str(zone) + ')')
        #                                 print('\nprocessing zone ' + str(czone) +
        #                                            ' of ' + str(nzones) + ' (' + str(zone) + ')')
        #                                 if zone not in zonestation: continue
        #                                 fcstzone[zone] = \
        #                                     forecast_zone(predictor=predictor,
        #                                                   param=param, predictorMonth=predictorMonth,
        #                                                   fcstPeriodType=config['fcstPeriodType'],
        #                                                   zone=zone, zonestation=zonestation, station_data_all=input_data,
        #                                                   sst_arr=sst_arr, lats=lats, lons=lons,
        #                                                   trainStartYear=config['trainStartYear'],
        #                                                   trainEndYear=config['trainEndYear'],
        #                                                   predictorStartYr=predictorStartYr,
        #                                                   fcstYear=config['fcstyear'], fcstPeriod=fcstPeriod,
        #                                                   PValue=config['PValue'],
        #                                                   composition=config.get('composition'), selectMode=config.get('selectMode'),
        #                                                   includeScore=config.get('includeScore'), stepwisePvalue=config.get('stepwisePvalue'),
        #                                                   outDir=outpath)
        #
        #                             window.progresslabel.setText('Finalizing: writing zone forecast')
        #                             print('Finalizing: writing zone forecast')
        #                             prefix = predictorName + '_' + param + '_' + predictorMonth
        #                             write_zone_forecast(prefix, zonejson, zoneattr, fcstzone, zones, outpath)
        #                             window.progresslabel.setText(cstatus + algorithm + ' complete')
        #                             print(algorithm + ' complete')
        #
        #                     # if no zone file has been defined...
        #                     else:
        #                         cstation = 0
        #                         for n in range(nstations):
        #                             cstation = cstation + 1
        #                             window.progresslabel.setText(cstatus + 'processing station ' + str(cstation) +
        #                                            ' of ' + str(nstations) + ' (' + str(stations[n]) + ')')
        #                             print('\nprocessing station ' + str(cstation) + ' of ' + str(nstations) +
        #                                   ' (' + str(stations[n]) + ')')
        #                             station_data_all = input_data.loc[input_data['ID'] == stations[n]]
        #                             lat[n] = station_data_all['Lat'].unique()[0]
        #                             lon[n] = station_data_all['Lon'].unique()[0]
        #                             qt1[n], qt2[n], qt3[n], prmean[n], fcst_precip[n], fcst_class[n], skills[n], probabilities[n] = \
        #                                 forecast_point(predictor=predictor,
        #                                            param=param, predictorMonth=predictorMonth, fcstPeriodType=config.get('fcstPeriodType'),
        #                                            station=stations[n], station_data_all=station_data_all,
        #                                            sst_arr=sst_arr, lats=lats, lons=lons, trainStartYear=config['trainStartYear'],
        #                                            trainEndYear=config['trainEndYear'], predictorStartYr=predictorStartYr,
        #                                            fcstYear=config['fcstyear'], fcstPeriod=fcstPeriod, PValue=config['PValue'],
        #                                            composition=config.get('composition'), selectMode=config.get('selectMode'),
        #                                            includeScore=config.get('includeScore'), stepwisePvalue=config.get('stepwisePvalue'),
        #                                            outDir=outpath)
        #
        #                         window.progresslabel.setText('Finalizing: writing station forecast')
        #                         print('Finalizing: writing station forecast')
        #                         prefix = predictorName + '_' + param + '_' + predictorMonth
        #                         write_forecast(prefix, lon, lat, stations, qt1, qt2, qt3, prmean, fcst_precip, fcst_class,
        #                                        skills, probabilities, outpath)
        #                         window.progresslabel.setText(cstatus + algorithm + ' complete')
        #                         print(algorithm+' complete')
        #             if algorithm == 'ANN':
        #



    # Set default values
    populate_period_list(config.get('fcstPeriodType'), config.get('fcstPeriodIndex'))
    window.startyearLineEdit.setText(str(config.get('trainStartYear')))
    window.endyearLineEdit.setText(str(config.get('trainEndYear')))
    window.predictMonthComboBox.setCurrentIndex(int(config.get('predictorMonthIndex',0)))
    window.inclscorecheckBox.setChecked(config.get('includeScore'))
    window.LRcheckBox.setChecked(config.get('enableLR'))
    window.pvaluelineEdit.setText(str(config.get('PValue')))
    window.swpvaluelineEdit.setText(str(config.get('stepwisePvalue')))
    window.neuronslineEdit.setText(str(config.get('neurons')))
    window.groupBox_5.setEnabled(True)
    window.missingvalueslineEdit.setText(str(config.get('predictantMissingValue')))
    window.outdirlabel.setText(config.get('outDir'))
    window.fcstyearlineEdit.setText(str(config.get('fcstyear')))
    window.zonevectorlabel.setText(config.get('zonevector',{}).get('file',''))
    for xx in config.get('zonevector', {}).get('attr',[]):
        window.zoneIDcomboBox.addItem(str(xx))
    window.zoneIDcomboBox.setCurrentIndex(config.get('zonevector', {}).get('ID',0))
    window.predictorlistWidget.clear()
    for fileName in config.get('predictorList'):
        window.predictorlistWidget.addItem(os.path.basename(fileName))
    window.predictantlistWidget.clear()
    for fileName in config.get('predictantList'):
        window.predictantlistWidget.addItem(os.path.basename(fileName))
    if config.get('inputFormat') == "CSV":
        window.CSVRadio.setChecked(True)
    else:
        window.NetCDFRadio.setChecked(True)
    if config.get('composition') == "Cumulation":
        window.cumRadio.setChecked(True)
    if config.get('composition') == "Average":
        window.avgRadio.setChecked(True)
    if 'LR' in config.get('algorithms'): window.LRcheckBox.setChecked(True)
    if 'LSTM' in config.get('algorithms'): window.LSTMcheckBox.setChecked(True)
    if 'MLP' in config.get('algorithms'): window.MLPcheckBox.setChecked(True)
    if 'CNN' in config.get('algorithms'): window.CNNcheckBox.setChecked(True)
    window.minlatLineEdit.setText(str(config.get("basinbounds",{}).get('minlat')))
    window.maxlatLineEdit.setText(str(config.get("basinbounds",{}).get('maxlat')))
    window.minlonLineEdit.setText(str(config.get("basinbounds",{}).get('minlon')))
    window.maxlonLineEdit.setText(str(config.get("basinbounds",{}).get('maxlon')))

    def closeapp():
        sys.exit(app.exec_())

    ## Signals
    window.outputButton.clicked.connect(getOutDir)
    window.period1Radio.toggled.connect(change_period_list)
    window.period2Radio.toggled.connect(change_period_list)
    window.addpredictButton.clicked.connect(addPredictors)
    window.removepredictButton.clicked.connect(removePredictors)
    window.browsepredictantButton.clicked.connect(addPredictants)
    window.clearpredictantButton.clicked.connect(clearPredictants)
    window.CSVRadio.toggled.connect(setInputFormat)
    window.inclscorecheckBox.toggled.connect(setInclude)
    window.ZoneButton.clicked.connect(addZoneVector)
    window.runButton.clicked.connect(launch_forecast_Thread)
    window.stopButton.clicked.connect(closeapp)
    sys.exit(app.exec_())
