# SADC Climate Services Centre
# SARCIS-DR Project
# Programming: Thembani Moitlhobogi
# Theory and Formulas: Sunshine Gamedze, Dr Arlindo Meque, Climate Experts from SADC NHMSs
# July 2020
#
import os, sys, re
from functions import *
from dateutil.relativedelta import relativedelta
from datetime import date, datetime
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import geojson
import threading

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QThread, QObject, QDate, QTime, QDateTime, Qt
# from citmain import *
qtCreatorFile = "CIT.ui"

# Global Variables
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
seasons = ['JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ','DJF']
csvheader = 'Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec'

outDir = ''
predictorList = []
predictantList = []
predictantMissingValue = -9999
cyear = QDate.currentDate().year()
fcstPeriodType = 0
fcstPeriodIndex = 9
trainStartYear = 1981
trainEndYear = 2010
predictorMonthIndex = 5
enableLR = True
PValue = 0.05
selectMode = 1
stepwisePvalue = 0.3
neurons = 20
inputFormat = "CSV"
includeScore = False
composition = "Cumulation"
zonevector = None
zoneID = None

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
        global outDir
        outDir = QtWidgets.QFileDialog.getExistingDirectory()
        window.outdirlabel.setText(outDir)

    def addPredictors():
        global predictorList
        fileNames = QtWidgets.QFileDialog.getOpenFileNames(window,
                    'Add File(s)', './', filter="NetCDF File (*.nc*)")
        for fileName in fileNames[0]:
            predictorList.append(fileName)
            window.predictorlistWidget.addItem(os.path.basename(fileName))

    def removePredictors():
        global predictorList
        newList = []
        for yy in predictorList:
            if os.path.basename(yy) != window.predictorlistWidget.selectedItems()[0].text():
                newList.append(yy)
        window.predictorlistWidget.clear()
        predictorList = newList
        for yy in newList:
            window.predictorlistWidget.addItem(os.path.basename(yy))

    def addPredictants():
        global inputFormat
        global predictantList
        global csvheader
        predictantList = []
        window.predictantlistWidget.clear()
        if inputFormat == "CSV":
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
                predictantList.append(filename)
                window.predictantlistWidget.addItem(os.path.basename(filename))
        else:
            fileName = QtWidgets.QFileDialog.getOpenFileName(window,
                    'Add File', './', filter="NetCDF File (*.nc*)")
            predictantList.append(fileName[0])
            window.predictantlistWidget.addItem(os.path.basename(fileName[0]))

    def clearPredictants():
        global predictantList
        predictantList = []
        window.predictantlistWidget.clear()

    def change_period_list():
        global fcstPeriodType
        periodlist = []
        window.periodComboBox.clear()
        if window.period1Radio.isChecked() == True:
            fcstPeriodType = 0
            periodlist = seasons
        if window.period2Radio.isChecked() == True:
            fcstPeriodType = 1
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
        global zonevector
        window.zoneIDcomboBox.clear()
        zonevector = None
        zonefieldsx = []
        window.zoneIDcomboBox.setDuplicatesEnabled(False)
        fileName = QtWidgets.QFileDialog.getOpenFileName(window,
                  'Add File', './', filter="GeoJson File (*.geojson)")
        zonevector = fileName[0]
        if os.path.isfile(zonevector):
            with open(zonevector) as f:
                zonejson = geojson.load(f)
            for zonekey in zonejson['features']:
                for zonetype in zonekey.properties:
                    zonefieldsx.append(zonetype)
            zonefields = []
            [zonefields.append(x) for x in zonefieldsx if x not in zonefields]
            for xx in zonefields:
                window.zoneIDcomboBox.addItem(str(xx))
            window.zonevectorlabel.setText(os.path.basename(zonevector))

    def setInputFormat():
        global inputFormat
        if window.CSVRadio.isChecked():
            inputFormat = "CSV"
        else:
            inputFormat = "NetCDF"

    def setInclude():
        global includeScore
        if window.inclscorecheckBox.isChecked():
            includeScore = True
        else:
            includeScore = False

    for xx in range(len(months)):
        window.predictMonthComboBox.addItem(months[xx])

    def launch_forecast_Thread():
        t = threading.Thread(target=forecast)
        t.start()

    def forecast():
        global composition
        global neurons
        global zonevector
        global stepwisePvalue
        window.progresslabel.setText('preparing inputs')
        algorithms = []
        if window.LRcheckBox.isChecked():
            algorithms.append('LR')
        if window.NNcheckBox.isChecked():
            if window.LSTMcheckBox.isChecked(): algorithms.append('LSTM')
            if window.MLPcheckBox.isChecked(): algorithms.append('MLP')
            if window.CNNcheckBox.isChecked(): algorithms.append('CNN')
            neurons = int(window.neuronslineEdit.text())

        if window.cumRadio.isChecked():
            composition = "Cumulation"
        else:
            composition = "Average"

        if window.period1Radio.isChecked():
            fcstPeriodType = 0
            fcstPeriodIndex = seasons.index(str(window.periodComboBox.currentText()))
        else:
            fcstPeriodType = 1
            fcstPeriodIndex = months.index(str(window.periodComboBox.currentText()))

        stepwisePvalue = float(window.swpvaluelineEdit.text())

        args = {
            'outDir': str(window.outdirlabel.text()),
            'cyear': int(window.fcstyearlineEdit.text()),
            'fcstPeriodType': fcstPeriodType,
            'fcstPeriodIndex': fcstPeriodIndex,
            'algorithms': algorithms,
            'enableLR': window.LRcheckBox.isChecked(),
            'PValue': float(window.pvaluelineEdit.text()),
            'neurons': int(window.neuronslineEdit.text()),
            'trainStartYear': int(window.startyearLineEdit.text()),
            'trainEndYear': int(window.endyearLineEdit.text()),
            'predictorMonthIndex': predictorMonthIndex,
            'predictorList': predictorList,
            'inputFormat': inputFormat,
            'composition': composition,
            'selectMode': selectMode,
            'zonevector': zonevector,
            'zoneID': str(window.zoneIDcomboBox.currentText()),
            'predictantList': predictantList,
            'predictantMissingValue': float(window.missingvalueslineEdit.text()),
        }
        # check if output directory exists
        if not os.path.exists(args['outDir']):
            window.progresslabel.setText("Output Directory not set!")
            return

        processes = len(args['predictorList']) * len(args['algorithms'])

        # prepare input data
        nstations = 0
        if args['inputFormat'] == 'CSV':
            input_df = pd.DataFrame()
            if len(predictantList) != 0:
                missing = args['predictantMissingValue']
                if len(str(missing)) == 0: missing = -9999
                input_data = concat_csvs(args['predictantList'], missing)
                stations = np.asarray(input_data['ID'].unique())
                nstations = stations.shape[0]
                fcst_precip = np.zeros(nstations)
                fcst_class = np.zeros(nstations)
                qt1 = np.zeros(nstations)
                qt2 = np.zeros(nstations)
                qt3 = np.zeros(nstations)
                prmean = np.zeros(nstations)
                lat = np.zeros(nstations)
                lon = np.zeros(nstations)
                skills = np.zeros(nstations)
                probabilities = [None] * nstations
            else:
                input_data = None

        predictorEndYr = args['cyear']
        predictorStartYr = args['trainStartYear']
        predictorMonth = str(window.predictMonthComboBox.currentText())
        fcstPeriod = str(window.periodComboBox.currentText())

        cproc = 0
        for predictor in args['predictorList']:
            if os.path.isfile(predictor):
                predictorName = os.path.splitext(os.path.basename(predictor))[0]
                window.progresslabel.setText('processing ' + predictorName)
                print('processing ' + predictorName)
                dataset = Dataset(predictor)
                sstmon = month_dict.get(predictorMonth.lower(), None)
                ref_date = str(dataset.variables['T'].units).split("since ", 1)[1]
                ref_date = datetime.strptime(ref_date, '%Y-%m-%d')
                param = get_parameter(dataset.variables.keys())
                timearr = np.array(dataset.variables['T'][:], dtype=int)
                sst = dataset.variables[param][:]
                lats = dataset.variables['Y'][:]
                lons = dataset.variables['X'][:]
                ncolssst = len(lons)
                nrowssst = len(lats)
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

                if args['predictorMonthIndex'] >= args['fcstPeriodIndex']:
                    predictorStartYr = args['trainStartYear'] - 1
                    predictorEndYr = args['cyear'] - 1

                if args['cyear'] > max(year_arr):
                    predictorStartYr = args['trainStartYear'] - 1
                    predictorEndYr = args['cyear'] - 1
                    if  args['cyear'] - max(year_arr) > 1:
                        status = "Predictor ("+param+") for " + predictorMonth + " goes up to " + str(year_arr[-1]) + \
                            ", cannot be used to forecast " + str(args['cyear']) + ' ' + fcstPeriod
                        window.progresslabel.setText(status)
                        continue
                    if args['fcstPeriodIndex'] >= args['predictorMonthIndex']:
                        status = "Predictor ("+param+") for " + predictorMonth + " goes up to " + str(year_arr[-1]) + \
                            ", cannot be used to forecast " + str(args['cyear']) + ' ' + fcstPeriod
                        window.progresslabel.setText(status)
                        continue

                if args['cyear'] <= args['trainEndYear']:
                    status = "Cannot forecast " + str(args['cyear']) + " as it is not beyond training period"
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
                yearspredictor = [yr for yr in range(predictorStartYr, predictorEndYr+1)]
                nsst_years = predictorEndYr - predictorStartYr + 1
                sst_arr = np.zeros((nsst_years, len(lats), len(lons))) * np.nan
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

                for algorithm in args['algorithms']:
                    cproc = cproc + 1
                    outpath = outDir + os.sep + 'Forecast_' + str(args['cyear']) + '_' + fcstPeriod + os.sep + algorithm
                    cstatus = '['+str(cproc)+'/'+str(processes)+'] '
                    if algorithm == 'LR':
                        if args['inputFormat'] == 'CSV':
                            # if a zone file has been defined...
                            zonevector = args['zonevector']
                            if zonevector is not None:
                                if os.path.isfile(zonevector):
                                    zoneID = args['zoneID']
                                    with open(zonevector) as f:
                                        zonejson = geojson.load(f)
                                    zones = zonelist(zonejson, zoneID)
                                    zonestation = {}
                                    fcstzone = {}
                                    # --------------
                                    for n in range(nstations):
                                        station_data_all = input_data.loc[input_data['ID'] == stations[n]]
                                        y = station_data_all['Lat'].unique()[0]
                                        z = station_data_all['Lon'].unique()[0]
                                        szone = whichzone(zonejson, y, z, zoneID)
                                        try:
                                            zonestation[szone].append(stations[n])
                                        except KeyError:
                                            zonestation[szone] = [stations[n]]
                                    window.progresslabel.setText(cstatus + "Calculating forecast for each zone...")
                                    nzones = len(zones)
                                    czone = 0
                                    for zone in zones:
                                        czone = czone + 1
                                        window.progresslabel.setText(cstatus + 'processing zone ' + str(czone) +
                                                   ' of ' + str(nzones) + ' (' + str(zone) + ')')
                                        print('\nprocessing zone ' + str(czone) +
                                                   ' of ' + str(nzones) + ' (' + str(zone) + ')')
                                        if zone not in zonestation: continue
                                        fcstzone[zone] = \
                                            forecast_zone(predictor=predictor,
                                                          param=param, predictorMonth=predictorMonth,
                                                          fcstPeriodType=fcstPeriodType,
                                                          zone=zone, zonestation=zonestation, station_data_all=input_data,
                                                          sst_arr=sst_arr, lats=lats, lons=lons,
                                                          trainStartYear=args['trainStartYear'],
                                                          trainEndYear=args['trainEndYear'],
                                                          predictorStartYr=predictorStartYr,
                                                          fcstYear=args['cyear'], fcstPeriod=fcstPeriod,
                                                          PValue=args['PValue'],
                                                          composition=args['composition'], selectMode=selectMode,
                                                          includeScore=includeScore, stepwisePvalue=stepwisePvalue,
                                                          outDir=outpath)

                                    window.progresslabel.setText('Finalizing: writing zone forecast')
                                    print('Finalizing: writing zone forecast')
                                    prefix = predictorName + '_' + param + '_' + predictorMonth
                                    write_zone_forecast(prefix, zonejson, zoneID, fcstzone, zones, outpath)
                                    window.progresslabel.setText(cstatus + algorithm + ' complete')
                                    print(algorithm + ' complete')

                            # if no zone file has been defined...
                            else:
                                cstation = 0
                                for n in range(nstations):
                                    cstation = cstation + 1
                                    window.progresslabel.setText(cstatus + 'processing station ' + str(cstation) +
                                                   ' of ' + str(nstations) + ' (' + str(stations[n]) + ')')
                                    print('\nprocessing station ' + str(cstation) + ' of ' + str(nstations) +
                                          ' (' + str(stations[n]) + ')')
                                    station_data_all = input_data.loc[input_data['ID'] == stations[n]]
                                    lat[n] = station_data_all['Lat'].unique()[0]
                                    lon[n] = station_data_all['Lon'].unique()[0]
                                    qt1[n], qt2[n], qt3[n], prmean[n], fcst_precip[n], fcst_class[n], skills[n], probabilities[n] = \
                                        forecast_point(predictor=predictor,
                                                   param=param, predictorMonth=predictorMonth, fcstPeriodType=fcstPeriodType,
                                                   station=stations[n], station_data_all=station_data_all,
                                                   sst_arr=sst_arr, lats=lats, lons=lons, trainStartYear=args['trainStartYear'],
                                                   trainEndYear=args['trainEndYear'], predictorStartYr=predictorStartYr,
                                                   fcstYear=args['cyear'], fcstPeriod=fcstPeriod, PValue=args['PValue'],
                                                   composition=args['composition'], selectMode=selectMode,
                                                   includeScore=includeScore, stepwisePvalue=stepwisePvalue,
                                                   outDir=outpath)

                                window.progresslabel.setText('Finalizing: writing station forecast')
                                print('Finalizing: writing station forecast')
                                prefix = predictorName + '_' + param + '_' + predictorMonth
                                write_forecast(prefix, lon, lat, stations, qt1, qt2, qt3, prmean, fcst_precip, fcst_class,
                                               skills, probabilities, outpath)
                                window.progresslabel.setText(cstatus + algorithm + ' complete')
                                print(algorithm+' complete')




    # Set default values
    populate_period_list(fcstPeriodType, fcstPeriodIndex)
    window.startyearLineEdit.setText(str(trainStartYear))
    window.endyearLineEdit.setText(str(trainEndYear))
    window.predictMonthComboBox.setCurrentIndex(predictorMonthIndex)
    window.inclscorecheckBox.setChecked(includeScore)
    window.LRcheckBox.setChecked(enableLR)
    window.pvaluelineEdit.setText(str(PValue))
    window.swpvaluelineEdit.setText(str(stepwisePvalue))
    window.neuronslineEdit.setText(str(neurons))
    window.groupBox_5.setEnabled(False)
    window.missingvalueslineEdit.setText(str(predictantMissingValue))

    if inputFormat == "CSV":
        window.CSVRadio.setChecked(True)
    else:
        window.NetCDFRadio.setChecked(True)
    if composition == "Cumulation":
        window.cumRadio.setChecked(True)
    else:
        window.avgRadio.setChecked(True)

    window.fcstyearlineEdit.setText(str(cyear))

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
