# SADC Climate Services Centre
# SARCIS-DR Project
# Programmer: Thembani Moitlhobogi
# Theory and Formulas: Sunshine Gamedze, Dr Arlindo Meque, Climate Experts from SADC NHMSs
# 10 August 2020
#
import os, sys, time, threading
from dateutil.relativedelta import relativedelta
from datetime import datetime
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import geojson, json
from multiprocessing import Pool, cpu_count
from functools import partial
from functions import *

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QThread, QObject, QDate, QTime, QDateTime, Qt
pwd = os.path.dirname(os.path.realpath('__file__'))
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
cpus = int(round(0.9 * cpu_count() - 0.5, 0))

#
def concat_csvs(csvs, missing):
    dfs_files = []
    for file in csvs:
        dfs_files.append(pd.read_csv(file, encoding = 'ISO-8859-9'))
    dfs_files = pd.concat((dfs_files), axis=0)
    dfs_files = dfs_files.replace(missing, np.nan)
    dfs_files = dfs_files.dropna(how='all')
    dfs_files['ID'] = dfs_files['ID'].apply(rename)
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

    try:
        with open(settingsfile, "r") as read_file:
            config = json.load(read_file)
    except:
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
        config['inputFormat'] = "CSV"
        config['composition'] = "Cumulation"
        config['zonevector'] = {"file": "", "ID": 0, "attr": []}
        config['fcstyear'] = fcstyear
        config['algorithms'] = []
        config['basinbounds'] = {"minlat": -90, "maxlat": 90, "minlon": -180, "maxlon": 366}
        config['plots'] = {'basemap': 'data\sadc_countries.geojson', 'zonepoints': 0}
        config['colors'] = {'class0': '#ffffff', 'class1': '#d2b48c', 'class2': '#fbff03', 'class3': '#0bfffb',
                            'class4': '#1601fc'}
        window.progresslabel.setText("Default settings loaded.")

    # def print_est_time():
    #     global config
    #     global cpus
    #     npredictors = len(config['predictorList'])
    #     nstations = len(predictantdict['stations'])
    #     ncpus = cpus
    #     window.progresslabel.setText(
    #         "Format error in " + os.path.basename(filename) + ", check if comma delimited")

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
        if len(window.predictorlistWidget.selectedItems()) == 0:
            return
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
        global cpus
        global pwd
        window.progresslabel.setText('preparing inputs')
        start_time = time.time()
        config['algorithms'] = []
        if window.LRcheckBox.isChecked():
            config['algorithms'].append('LR')
        if window.MLPcheckBox.isChecked(): config['algorithms'].append('MLP')

        if len(config.get('algorithms')) == 0:
            window.progresslabel.setText("No algorithm set!")
            return None

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
        config['basinbounds']['minlat'] = float(str(window.minlatLineEdit.text()).strip() or -90)
        config['basinbounds']['maxlat'] = float(str(window.maxlatLineEdit.text()).strip() or 90)
        config['basinbounds']['minlon'] = float(str(window.minlonLineEdit.text()).strip() or -180)
        config['basinbounds']['maxlon'] = float(str(window.maxlonLineEdit.text()).strip() or 366)
        config['trainStartYear'] = int(window.startyearLineEdit.text())
        config['trainEndYear'] = int(window.endyearLineEdit.text())

        # check if output directory exists
        if not os.path.exists(config.get('outDir')):
            window.progresslabel.setText("Output Directory not set!")
            return None

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
                predictantdict['lats'], predictantdict['lons'] = [], []
                for n in range(nstations):
                    station_data_all = input_data.loc[input_data['ID'] == stations[n]]
                    predictantdict['lats'].append(station_data_all['Lat'].unique()[0])
                    predictantdict['lons'].append(station_data_all['Lon'].unique()[0])

            else:
                input_data = None

        predictorEndYr = int(config.get('fcstyear'))
        predictorStartYr = int(config.get('trainStartYear'))
        predictorMonth = str(window.predictMonthComboBox.currentText())
        fcstPeriod = str(window.periodComboBox.currentText())

        for predictor in config.get('predictorList'):
            if os.path.isfile(predictor):
                predictorName = os.path.splitext(os.path.basename(predictor))[0]
                predictordict[predictorName] = {}
                window.progresslabel.setText('checking ' + predictorName)
                print('checking ' + predictorName)
                dataset = Dataset(predictor)
                sstmon = month_dict.get(predictorMonth.lower(), None)
                ref_date = str(dataset.variables['T'].units).split("since ", 1)[1]
                ref_date = datetime.strptime(ref_date, '%Y-%m-%d')
                param = get_parameter(dataset.variables.keys())
                timearr = np.array(dataset.variables['T'][:], dtype=int)
                sst = dataset.variables[param][:]
                sst[sst.mask] = np.nan
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
                    return None

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
                        return None
                    if config.get('fcstPeriodIndex') >= config.get('predictorMonthIndex'):
                        status = "Predictor ("+param+") for " + predictorMonth + " goes up to " + str(year_arr[-1]) + \
                            ", cannot be used to forecast " + str(config.get('fcstyear')) + ' ' + fcstPeriod
                        window.progresslabel.setText(status)
                        return None

                if int(config.get('fcstyear')) <= int(config.get('trainEndYear')):
                    status = "Cannot forecast " + str(config.get('fcstyear')) + " as it is not beyond training period"
                    window.progresslabel.setText(status)
                    return None

                if predictorStartYr < year_arr[0]:
                    status = "Predictor ("+param+") data starts in " + str(year_arr[0]) + \
                        ", selected options require predictor to start in " + str(predictorStartYr)
                    window.progresslabel.setText(status)
                    return None

                status = 'predictor data to be used: ' + str(predictorStartYr) + predictorMonth + ' to ' + \
                         str(predictorEndYr) + predictorMonth
                window.progresslabel.setText(status)
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

        # create output directory
        outdir = config.get('outDir') + os.sep + 'Forecast_' + str(config.get('fcstyear')) + \
                 '_' + fcstPeriod + os.sep
        os.makedirs(outdir, exist_ok=True)

        # print(forecast_station(config, predictordict, predictantdict, fcstPeriod, stations[0]))
        func = partial(forecast_station, config, predictordict, predictantdict, fcstPeriod, outdir)
        p = Pool(cpus)
        rs = p.imap_unordered(func, stations)
        p.close()
        while (True):
            completed = rs._index
            status = "completed processing " + str(completed) + " of " + str(len(stations)) + " stations"
            window.progresslabel.setText(status)
            if (completed >= len(stations)): break
            # print("processing ", completed, " of ", len(stations))
            time.sleep(0.2)
        outs = list(rs)
        outputs = []
        for out in outs:
            if isinstance(out, pd.DataFrame):
                if out.shape[0] > 0:
                    outputs.append(out)
        if len(outputs) == 0:
            window.progresslabel.setText('Skill not enough to produce forecast')
            print('Skill not enough to produce forecast')
        else:
            # Write forecasts to output directory
            forecastsdf = pd.concat(outputs, ignore_index=True)
            window.progresslabel.setText('Writing Forecast...')
            print('Writing Forecast...')
            forecastdir = outdir + os.sep + "Forecast"
            os.makedirs(forecastdir, exist_ok=True)
            fcstprefix = str(config.get('fcstyear')) + fcstPeriod + '_' + predictordict[predictorName]['predictorMonth']
            colors = config.get('colors', {})
            fcstName = str(config.get('fcstyear')) + fcstPeriod
            # write forecast by station or zone
            if len(config.get('zonevector', {}).get('file')) == 0:
                fcstcsvout = forecastdir + os.sep + fcstprefix + '_forecast.csv'
                forecastsdf.to_csv(fcstcsvout, header=True, index=True)
                highskilldf = forecastsdf[forecastsdf.HS.ge(50)][['ID', 'Lat', 'Lon', 'HS', 'class']]
                r, _ = highskilldf.shape
                if r > 0:
                    stationclass = highskilldf.groupby(['ID', 'Lat', 'Lon']).apply(func=weighted_average).to_frame(name='WA')
                    stationclass[['wavg', 'class4', 'class3', 'class2', 'class1']] = pd.DataFrame(stationclass.WA.tolist(), index=stationclass.index)
                    stationclass = stationclass.drop(['WA'], axis=1)
                    stationclass['class'] = (stationclass['wavg']+0.5).astype(int)
                    stationclassout = forecastdir + os.sep + fcstprefix + '_station-forecast.csv'
                    stationclass.to_csv(stationclassout, header=True, index=True)
                    fcstjsonout = forecastdir + os.sep + fcstprefix + '_station-forecast.geojson'
                    stationclass = stationclass.reset_index()
                    data2geojson(stationclass, fcstjsonout)
                    base_map = None
                    base_mapfile = config.get('plots', {}).get('basemap', '')
                    if not os.path.isfile(base_mapfile):
                        base_mapfile = repr(config.get('plots', {}).get('basemap'))
                    if os.path.isfile(base_mapfile):
                        with open(base_mapfile, "r") as read_file:
                            base_map = geojson.load(read_file)
                    station_forecast_png(fcstprefix, stationclass, base_map, colors, forecastdir, fcstName)
                    window.progresslabel.setText('Done in '+str(convert(time.time()-start_time)))
                    print('Done in '+str(convert(time.time()-start_time)))
                else:
                    window.progresslabel.setText('Skill not enough for station forecast')
                    print('Skill not enough for station forecast')
            else:
                if not os.path.isfile(config.get('zonevector', {}).get('file')):
                    window.progresslabel.setText('Error: Zone vector does not exist, will not write zone forecast')
                    print('Error: Zone vector does not exist, will not write zone forecast')
                else:
                    with open(config.get('zonevector', {}).get('file')) as f:
                            zonejson = geojson.load(f)
                    zoneattrID = config.get('zonevector',{}).get('ID')
                    zoneattr = config.get('zonevector', {}).get('attr')[zoneattrID]
                    forecastsdf["Zone"] = np.nan
                    # --------------
                    for n in range(nstations):
                        station = predictantdict['stations'][n]
                        szone = whichzone(zonejson, predictantdict['lats'][n], predictantdict['lons'][n], zoneattr)
                        forecastsdf.loc[forecastsdf.ID == station, 'Zone'] = szone
                    fcstcsvout = forecastdir + os.sep + fcstprefix + '_zone_station_forecast.csv'
                    forecastsdf.to_csv(fcstcsvout, header=True, index=True)
                    # generate zone forecast
                    zonefcstprefix = forecastdir + os.sep + str(config.get('fcstyear')) + fcstPeriod + '_' + \
                                     predictordict[predictorName]['predictorMonth']
                    highskilldf = forecastsdf[forecastsdf.HS.ge(50)][['HS', 'class', 'Zone']]
                    r, _ = highskilldf.shape
                    if r > 0:
                        stationsdf = forecastsdf[forecastsdf.HS.ge(50)][['ID', 'Lat', 'Lon', 'HS', 'class']]
                        stationclass = stationsdf.groupby(['ID', 'Lat', 'Lon']).apply(func=weighted_average).to_frame(
                            name='WA')
                        stationclass[['wavg', 'class4', 'class3', 'class2', 'class1']] = pd.DataFrame(
                            stationclass.WA.tolist(), index=stationclass.index)
                        stationclass = stationclass.drop(['WA'], axis=1)
                        stationclass['class'] = (stationclass['wavg']+0.5).astype(int)
                        zoneclass = highskilldf.groupby('Zone').apply(func=weighted_average).to_frame(name='WA')
                        zoneclass[['wavg', 'class4', 'class3', 'class2', 'class1']] = pd.DataFrame(zoneclass.WA.tolist(), index=zoneclass.index)
                        zoneclass = zoneclass.drop(['WA'], axis=1)
                        zoneclass['class'] = (zoneclass['wavg']+0.5).astype(int)
                        ZoneID = config['zonevector']['attr'][config['zonevector']['ID']]
                        zonepoints = config.get('plots', {}).get('zonepoints', '0')
                        write_zone_forecast(zonefcstprefix, zoneclass, zonejson, ZoneID, colors, stationclass, zonepoints,
                                            fcstName)
                        window.progresslabel.setText('Done in '+str(convert(time.time()-start_time)))
                        print('Done in '+str(convert(time.time()-start_time)))
                    else:
                        window.progresslabel.setText('Skill not enough for zone forecast')
                        print('Skill not enough for zone forecast')


    # Set default values
    populate_period_list(config.get('fcstPeriodType'), config.get('fcstPeriodIndex'))
    window.startyearLineEdit.setText(str(config.get('trainStartYear')))
    window.endyearLineEdit.setText(str(config.get('trainEndYear')))
    window.predictMonthComboBox.setCurrentIndex(int(config.get('predictorMonthIndex',0)))
    window.LRcheckBox.setChecked(config.get('enableLR'))
    window.pvaluelineEdit.setText(str(config.get('PValue')))
    window.swpvaluelineEdit.setText(str(config.get('stepwisePvalue')))
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
    if 'MLP' in config.get('algorithms'): window.MLPcheckBox.setChecked(True)
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
    window.ZoneButton.clicked.connect(addZoneVector)
    window.runButton.clicked.connect(launch_forecast_Thread)
    window.stopButton.clicked.connect(closeapp)
    sys.exit(app.exec_())
