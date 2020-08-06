# SADC Climate Services Centre
# SARCIS-DR Project
# Programmer: Thembani Moitlhobogi
# Theory and Formulas: Sunshine Gamedze, Dr Arlindo Meque, Climate Experts from SADC NHMSs
# 23 August 2019
#
import os, sys, re, time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import linear_model
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn import cluster as skluster
from shapely.geometry import shape, Point
# from itertools import combinations
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from osgeo import gdal
import warnings
import numpy as np
import geojson, json

SSTclusterSize=1200.
kms_per_radian = 6371.0088
epsilon = SSTclusterSize*1.0 / kms_per_radian

# --- constants ---
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
seasons = ['JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF']
month_dict = {'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06','jul':'07',
              'aug':'08','sep':'09','oct':'10','nov':'11','dec':'12'}
proj='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
     'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],' \
     'AUTHORITY["EPSG","4326"]]'

driver = gdal.GetDriverByName('GTiff')

# --- functions ---


def integer(x):
    if np.isfinite(x):
        return int(round(x,0))
    else:
        return np.nan


def season_cumulation(dfm, year, season):
    nyear = year + 1
    try:
        if season == 'JFM':
            if np.array(np.isfinite(dfm.loc[[year], 'Jan':'Mar'])).all(): return float(round(
                dfm.loc[[year], 'Jan':'Mar'].sum(axis=1), 1))
        if season == 'FMA':
            if np.array(np.isfinite(dfm.loc[[year], 'Feb':'Apr'])).all(): return float(round(
                dfm.loc[[year], 'Feb':'Apr'].sum(axis=1), 1))
        if season == 'MAM':
            if np.array(np.isfinite(dfm.loc[[year], 'Mar':'May'])).all(): return float(round(
                dfm.loc[[year], 'Mar':'May'].sum(axis=1), 1))
        if season == 'AMJ':
            if np.array(np.isfinite(dfm.loc[[year], 'Apr':'Jun'])).all(): return float(round(
                dfm.loc[[year], 'Apr':'Jun'].sum(axis=1), 1))
        if season == 'MJJ':
            if np.array(np.isfinite(dfm.loc[[year], 'May':'Jul'])).all(): return float(round(
                dfm.loc[[year], 'May':'Jul'].sum(axis=1), 1))
        if season == 'JJA':
            if np.array(np.isfinite(dfm.loc[[year], 'Jun':'Aug'])).all(): return float(round(
                dfm.loc[[year], 'Jun':'Aug'].sum(axis=1), 1))
        if season == 'JAS':
            if np.array(np.isfinite(dfm.loc[[year], 'Jul':'Sep'])).all(): return float(round(
                dfm.loc[[year], 'Jul':'Sep'].sum(axis=1), 1))
        if season == 'ASO':
            if np.array(np.isfinite(dfm.loc[[year], 'Aug':'Oct'])).all(): return float(round(
                dfm.loc[[year], 'Aug':'Oct'].sum(axis=1), 1))
        if season == 'SON':
            if np.array(np.isfinite(dfm.loc[[year], 'Sep':'Nov'])).all(): return float(round(
                dfm.loc[[year], 'Sep':'Nov'].sum(axis=1), 1))
        if season == 'OND':
            if np.array(np.isfinite(dfm.loc[[year], 'Oct':'Dec'])).all(): return float(round(
                dfm.loc[[year], 'Oct':'Dec'].sum(axis=1), 1))
        if season == 'NDJ':
            p1 = np.array(np.isfinite(dfm.loc[[year], 'Nov':'Dec'])).all()
            p2 = np.array(np.isfinite(dfm.loc[[nyear], 'Jan'])).all()
            if p1 and p2: return round(float(dfm.loc[[year], 'Nov':'Dec'].sum(axis=1)) + float(dfm.loc[[nyear], 'Jan']),
                                       1)
        if season == 'DJF':
            p1 = np.array(np.isfinite(dfm.loc[[year], 'Dec'])).all()
            p2 = np.array(np.isfinite(dfm.loc[[nyear], 'Jan':'Feb'])).all()
            if p1 and p2: return round(float(dfm.loc[[year], 'Dec']) + float(dfm.loc[[nyear], 'Jan':'Feb'].sum(axis=1)),
                                       1)
    except KeyError:
        return


def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out=0.4,
                       verbose=True):
    included = list(initial_list)
    comment = []
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed = True
            comment.append('Add {:4} with p-value {:.3}'.format(best_feature, best_pval))
        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            comment.append('Drop {:4} with p-value {:.3}'.format(worst_feature, worst_pval))

        if not changed:
            break
    return included, comment


def season_average(dfm,year,season):
  nyear=year+1
  try:
    if season=='JFM':
      if np.array(np.isfinite(dfm.loc[[year],'Jan':'Mar'])).all(): return float(round(dfm.loc[[year],'Jan':'Mar'].mean(axis=1),1))
    if season=='FMA':
      if np.array(np.isfinite(dfm.loc[[year],'Feb':'Apr'])).all(): return float(round(dfm.loc[[year],'Feb':'Apr'].mean(axis=1),1))
    if season=='MAM':
      if np.array(np.isfinite(dfm.loc[[year],'Mar':'May'])).all(): return float(round(dfm.loc[[year],'Mar':'May'].mean(axis=1),1))
    if season=='AMJ':
      if np.array(np.isfinite(dfm.loc[[year],'Apr':'Jun'])).all(): return float(round(dfm.loc[[year],'Apr':'Jun'].mean(axis=1),1))
    if season=='MJJ':
      if np.array(np.isfinite(dfm.loc[[year],'May':'Jul'])).all(): return float(round(dfm.loc[[year],'May':'Jul'].mean(axis=1),1))
    if season=='JJA':
      if np.array(np.isfinite(dfm.loc[[year],'Jun':'Aug'])).all(): return float(round(dfm.loc[[year],'Jun':'Aug'].mean(axis=1),1))
    if season=='JAS':
      if np.array(np.isfinite(dfm.loc[[year],'Jul':'Sep'])).all(): return float(round(dfm.loc[[year],'Jul':'Sep'].mean(axis=1),1))
    if season=='ASO':
      if np.array(np.isfinite(dfm.loc[[year],'Aug':'Oct'])).all(): return float(round(dfm.loc[[year],'Aug':'Oct'].mean(axis=1),1))
    if season=='SON':
      if np.array(np.isfinite(dfm.loc[[year],'Sep':'Nov'])).all(): return float(round(dfm.loc[[year],'Sep':'Nov'].mean(axis=1),1))
    if season=='OND':
      if np.array(np.isfinite(dfm.loc[[year],'Oct':'Dec'])).all(): return float(round(dfm.loc[[year],'Oct':'Dec'].mean(axis=1),1))
    if season=='NDJ':
        p1=np.array(np.isfinite(dfm.loc[[year],'Nov':'Dec'])).all()
        p2=np.array(np.isfinite(dfm.loc[[nyear],'Jan'])).all()
        if p1 and p2: return round((float(dfm.loc[[year],'Nov':'Dec'].sum(axis=1))+float(dfm.loc[[nyear],'Jan']))/3.,1)
    if season=='DJF':
        p1=np.array(np.isfinite(dfm.loc[[year],'Dec'])).all()
        p2=np.array(np.isfinite(dfm.loc[[nyear],'Jan':'Feb'])).all()
        if p1 and p2: return round((float(dfm.loc[[year],'Dec'])+float(dfm.loc[[nyear],'Jan':'Feb'].sum(axis=1)))/3.,1)
  except:
    return


def dbcluster(coordinates, func, n_clusters, mindist, samples, njobs):
    if func == 'kmeans':
        db = skluster.KMeans(n_clusters=n_clusters).fit(coordinates)
    if func == 'dbscan':
        db = DBSCAN(eps=mindist * 1.0 / 6371.0088, min_samples=samples, n_jobs=1)
        db = db.fit(np.radians(coordinates))
    return db


def data2geojson(dfw, jsonout):
    dfw = dfw.fillna('')
    features = []
    insert_features = lambda X: features.append(
        geojson.Feature(geometry=geojson.Point((X["Lon"],
                                                X["Lat"])),
                        properties=dict(predictor=X["Predictor"], algorithm=X["Algorithm"], ID=X["ID"], tercile1=X["t1"],
                                        tercile2=X["t2"], tercile3=X["t3"], mean=X["mean"], fcst_precip=X["fcst"],
                                        fcst_class=X["class"], r2score=X["r2score"], hitscore=X["HS"],
                                        PB_PN_PA=X["Prob"])))
    dfw.apply(insert_features, axis=1)
    with open(jsonout, 'w') as fp:
        geojson.dump(geojson.FeatureCollection(features), fp, sort_keys=False, ensure_ascii=False)


def whichzone(zonejson, lat, lon, field):
    point = Point(lon, lat)
    for feature in zonejson['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            return feature['properties'][field]
    return None


def zonelist(zonejson, field):
    zarr = []
    for feature in zonejson['features']:
        zarr.append(feature['properties'][field])
    return zarr

def write_forecast(prefix, forecast_df, outdir):
    fcstjsonout = outdir + os.sep + prefix + '_forecast.geojson'
    data2geojson(forecast_df,fcstjsonout)
    fcstcsvout = outdir + os.sep + prefix + '_forecast.csv'
    forecast_df.to_csv(fcstcsvout, header=True, index=True)

def write_zone_forecast(prefix, forecastjson, zoneID, fcstzone, zones, outpath):
    fcstzone_df = pd.DataFrame(columns=['ZoneID','t1','t2','t3','mean','fcst','class','hitscore','PB_PN_PA'])
    fcstzone_df['ZoneID'] = zones
    fcstzone_df.set_index('ZoneID', inplace=True)
    for zone in fcstzone.keys():
        fcstzone_df.loc[[zone], 't1'] = fcstzone[zone][0]
        fcstzone_df.loc[[zone], 't2'] = fcstzone[zone][1]
        fcstzone_df.loc[[zone], 't3'] = fcstzone[zone][2]
        fcstzone_df.loc[[zone], 'mean'] = fcstzone[zone][3]
        fcstzone_df.loc[[zone], 'fcst'] = fcstzone[zone][4]
        fcstzone_df.loc[[zone], 'class'] = fcstzone[zone][5]
        fcstzone_df.loc[[zone], 'hitscore'] = fcstzone[zone][6]
        fcstzone_df.loc[[zone], 'PB_PN_PA'] = str(fcstzone[zone][7]).replace('nan','')
    fcstzone_df = fcstzone_df.fillna('')
    fcstoutdir = outpath + os.sep + "forecast"
    os.makedirs(fcstoutdir, exist_ok=True)
    fcstjsonout = fcstoutdir + os.sep + prefix + '_zone-forecast.geojson'
    fcstcsvout = fcstoutdir + os.sep + prefix + '_zone-forecast.csv'
    for feature in forecastjson['features']:
        ID = feature['properties'][zoneID]
        if not fcstzone.get(ID, None) == None:
            feature['properties']['t1'] = list(fcstzone_df.loc[[ID],'t1'])[0]
            feature['properties']['t2'] = list(fcstzone_df.loc[[ID],'t2'])[0]
            feature['properties']['t3'] = list(fcstzone_df.loc[[ID],'t3'])[0]
            feature['properties']['mean'] = list(fcstzone_df.loc[[ID],'mean'])[0]
            feature['properties']['fcst'] = list(fcstzone_df.loc[[ID],'fcst'])[0]
            feature['properties']['class'] = list(fcstzone_df.loc[[ID],'class'])[0]
            feature['properties']['hitscore'] = list(fcstzone_df.loc[[ID],'hitscore'])[0]
            feature['properties']['PB_PN_PA'] = list(fcstzone_df.loc[[ID],'PB_PN_PA'])[0]
    fcstzone_df.to_csv(fcstcsvout, header=True, index=True)
    with open(fcstjsonout, 'w') as fp:
        geojson.dump(forecastjson, fp)


def model_skill(fcst_df, lim1, lim2):
    HS = 0
    HSS = 0
    POD_below = 0
    POD_normal = 0
    POD_above = 0
    FA_below = 0
    FA_normal = 0
    FA_above = 0
    cgtable_df = pd.DataFrame(columns=['-','FCST BELOW','FCST NORMAL','FCST ABOVE','Total'])
    cgtable_df['-'] = ['OBS BELOW','OBS_NORMAL','OBS_ABOVE','Total']
    df = fcst_df.dropna()
    obs = np.array(df.sort_values(by=['obs'])['obs'])
    fcst = np.array(df.sort_values(by=['obs'])['fcst'])
    below = obs <= lim1
    normal = (obs > lim1) & (obs <= lim2)
    above = obs > lim2
    A11 = sum(fcst[below] <= lim1)
    A12 = sum((fcst[below] > lim1) & (fcst[below] <= lim2))
    A13 = sum(fcst[below] > lim2)
    A21 = sum(fcst[normal] <= lim1)
    A22 = sum((fcst[normal] > lim1) & (fcst[normal] <= lim2))
    A23 = sum(fcst[normal] > lim2)
    A31 = sum(fcst[above] <= lim1)
    A32 = sum((fcst[above] > lim1) & (fcst[above] <= lim2))
    A33 = sum(fcst[above] > lim2)
    M = A11 + A21 + A31
    N = A12 + A22 + A32
    O = A13 + A23 + A33
    J = A11 + A12 + A13
    K = A21 + A22 + A23
    L = A31 + A32 + A33
    T = M + N + O
    cgtable_df['FCST BELOW'] = [A11, A21, A31, M]
    cgtable_df['FCST NORMAL'] = [A12, A22, A32, N]
    cgtable_df['FCST ABOVE'] = [A13, A23, A33, O]
    cgtable_df['Total'] = [J, K, L, T]
    if T != 0:
        Factor = (J*M + K*N + L*O) / T
        HS = integer(100 * (A11 + A22 + A33) / T)
        HSS = (A11 + A22 + A33 - Factor) / (T - Factor)
    if A11 != 0: POD_below = integer(100 * A11 / M)
    if A22 != 0: POD_normal = integer(100 * A22 / N)
    if A33 != 0: POD_above = integer(100 * A33 / O)
    if M != 0: FA_below = integer(100 * A31 / M)
    if N != 0: FA_normal = integer(100 * (A12 + A32) / N)
    if O != 0: FA_above = integer(100 * A13 / O)
    warnings.filterwarnings('error')
    try:
        PB = [POD_below, integer(100 * A21 / M), integer(100 * A31 / M)]
    except:
        PB = np.nan
    try:
        PN = [integer(100 * A12 / N), POD_normal, integer(100 * A32 / N)]
    except:
        PN = np.nan
    try:
        PA = [FA_above, integer(100 * A23 / O), POD_above]
    except:
        PA = np.nan
    return HS, HSS, POD_below, POD_normal, POD_above, FA_below, FA_normal, FA_above, cgtable_df, PB, PN, PA

def plot_Station_forecast(forecast_df, fcstPeriod, graphpng, station, q1, q2, q3):
    DPI = 100
    W = 1000
    H = 600
    colors = ['#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
              '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']
    graphs = list(forecast_df.columns)
    indx = graphs.index('Year')
    graphs.pop(indx)
    maxval = np.nanmax(np.ravel(forecast_df[graphs]))
    minval = np.nanmin(np.ravel(forecast_df[graphs]))
    indx = graphs.index(fcstPeriod)
    graphs.pop(indx)
    q1s = [q1] * len(forecast_df['Year'])
    q2s = [q2] * len(forecast_df['Year'])
    q3s = [q3] * len(forecast_df['Year'])
    maxvals = [maxval + (0.05 * maxval)] * len(forecast_df['Year'])
    minvals = [minval - abs(0.05 * minval)] * len(forecast_df['Year'])
    plt.figure(figsize=(W/float(DPI), H/float(DPI)), frameon=True, dpi=DPI)
    plt.fill_between(forecast_df['Year'], minvals, q1s, color='#ffe7d1')
    plt.fill_between(forecast_df['Year'], q1s, q3s, color='#e8f9e9')
    plt.fill_between(forecast_df['Year'], q3s, maxvals, color='#f4f6ff')
    # plt.plot(forecast_df['Year'], [q1] * len(list(forecast_df['Year'])), color='#e5e5e5', linestyle='dashed')
    plt.plot(forecast_df['Year'], [q2] * len(list(forecast_df['Year'])), color='#e5e5e5', linestyle='dashed')
    # plt.plot(forecast_df['Year'], [q3] * len(list(forecast_df['Year'])), color='#e5e5e5', linestyle='dashed')
    plt.plot(forecast_df['Year'], forecast_df[fcstPeriod], color='red', marker='o', label='Actual')
    for n in range(len(graphs)):
        graph = graphs[n]
        plt.plot(forecast_df['Year'], forecast_df[graph], color=colors[n], marker='+', label=graph, linewidth=0.7)
    plt.title('Actual ('+fcstPeriod+') vs Forecasts for '+station, fontsize=12)
    plt.legend(prop={'size': 6})
    plt.xticks(list(forecast_df['Year']), [str(x) for x in list(forecast_df['Year'])], fontsize=8)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Forecast', fontsize=12)
    plt.savefig(graphpng, bbox_inches = 'tight')


# def best_basins(all_basins, basin_matrix, actual, trainingYears, sstyears, includeScore):
#     combos = {}
#     regrFormula = {}
#     r2scores = []
#     ntest_actualyrs = len(trainingYears)
#     ntest_sstyears = len(trainingYears)
#     if includeScore:
#         ntest_actualyrs = len(actual)
#     r2scores_df = pd.DataFrame(columns=['r2_score', 'Basin combination'])
#     regr = linear_model.LinearRegression()
#     combinations_all = sum([list(map(list, combinations(all_basins, i))) for i in range(len(all_basins) + 1)], [])
#     for cmb in range(len(combinations_all)):
#         fcst = []
#         comb = combinations_all[cmb]
#         if len(comb) < 1: continue
#         combos.update( {cmb : comb} )
#         combo_basin_matrix = np.zeros((len(sstyears), len(comb))) * np.nan
#         # loop for all years where SST is available
#         for yr in range(len(sstyears)):
#             for group in range(len(comb)):
#                 # get corresponding sst average for the group from main basin_matrix
#                 combo_basin_matrix[yr][group] = basin_matrix[yr][all_basins.index(comb[group])]
#         training_combo_basin_matrix = combo_basin_matrix[:len(trainingYears)]
#         training_actual = actual[:len(trainingYears)]
#         notnull = np.isfinite(training_actual)
#         regr.fit(training_combo_basin_matrix[notnull], training_actual[notnull])
#         intercept = regr.intercept_
#         coefficients = regr.coef_
#         test_combo_basin_matrix = np.asarray(combo_basin_matrix)
#         test_actual = list(actual[:len(trainingYears)])
#         notnull = np.isfinite(test_actual)
#         for yr in range(len(sstyears)):
#             fcst.append(round(regr.predict([test_combo_basin_matrix[yr]])[0], 1))
#         r2score = r2_score(np.asarray(test_actual)[:len(trainingYears)][notnull], np.asarray(fcst)[:len(trainingYears)][notnull])
#         r2scores.append(r2score)
#         r2scores_df = r2scores_df.append({'r2_score': r2score, 'Basin combination': comb}, ignore_index=True)
#         if r2score >= max(r2scores):
#             final_basins = comb
#             final_basin_matrix = test_combo_basin_matrix
#             final_forecasts = np.zeros(len(fcst)) * np.nan
#             final_forecasts[np.array(fcst) >= 0] = np.array(fcst)[np.array(fcst) >= 0]
#             bestr2score = r2score
#             regrFormula = {"intercept": intercept, "coefficients": coefficients}
#     return bestr2score, final_basins, final_forecasts, final_basin_matrix, r2scores_df, regrFormula


# def best_basins_model(all_basins, basin_matrix, actual, trainingYears, sstyears, includeScore, stepwisePvalue):
#     ntest_actualyrs = len(trainingYears)
#     if includeScore:
#         ntest_actualyrs = len(actual)
#     basin_matrix_df = pd.DataFrame(list(basin_matrix[:len(trainingYears)]), columns=all_basins)
#     notnull = np.isfinite(np.array(actual[:len(trainingYears)]))
#     final_basins, comments = stepwise_selection(basin_matrix_df[notnull], list(actual[:len(trainingYears)][notnull]),
#                                                 initial_list=all_basins, threshold_out=stepwisePvalue)
#     comment_df = pd.DataFrame(columns=['Comment'])
#     comment_df['Comment'] = comments
#     r2scores_df = pd.DataFrame(columns=['r2_score', 'Basin combination'])
#     regr = linear_model.LinearRegression()
#     fcst = []
#     combo_basin_matrix = np.zeros((len(sstyears), len(final_basins))) * np.nan
#     # loop for all years where SST is available
#     for yr in range(len(sstyears)):
#         for group in range(len(final_basins)):
#             # get corresponding sst average for the group from main basin_matrix
#             combo_basin_matrix[yr][group] = basin_matrix[yr][all_basins.index(final_basins[group])]
#
#     training_combo_basin_matrix = combo_basin_matrix[:len(trainingYears)]
#     training_actual = actual[:len(trainingYears)]
#     notnull = np.isfinite(training_actual)
#     regr.fit(training_combo_basin_matrix[notnull], np.asarray(training_actual)[notnull])
#     intercept = regr.intercept_
#     coefficients = regr.coef_
#     regrFormula = {"intercept": intercept, "coefficients": coefficients}
#     test_actual = list(actual[:len(trainingYears)])
#     notnull = np.isfinite(test_actual)
#     for yr in range(len(sstyears)):
#         fcst.append(round(regr.predict([np.asarray(combo_basin_matrix)[yr]])[0], 1))
#     bestr2score = r2_score(np.asarray(actual[:len(trainingYears)])[notnull], np.asarray(fcst[:len(trainingYears)])[notnull])
#     r2scores_df = r2scores_df.append({'r2_score': bestr2score, 'Basin combination': final_basins}, ignore_index=True)
#     final_basin_matrix = combo_basin_matrix
#     final_forecasts = np.zeros(len(fcst)) * np.nan
#     final_forecasts[np.array(fcst) >= 0] = np.array(fcst)[np.array(fcst) >= 0]
#     return bestr2score, final_basins, final_forecasts, final_basin_matrix, r2scores_df, regrFormula, comment_df


def writeout(prefix, r_matrix, p_matrix, corgrp_matrix, corr_df, lats, lons, outdir):
    ncolssst = len(lons)
    nrowssst = len(lats)
    flats = lats[::-1]
    nx = len(lons)
    ny = len(flats)
    xmin, ymin, xmax, ymax = [lons.min(), flats.min(), lons.max(), flats.max()]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    os.makedirs(outdir, exist_ok=True)
    # write p-matrix image
    output1 = outdir + os.sep + prefix + '_correlation-r-values.tif'
    output = outdir + os.sep + prefix + '_correlation-p-values.tif'
    dataset1 = driver.Create(output1, ncolssst, nrowssst, eType=gdal.GDT_Float32)
    dataset2 = driver.Create(output, ncolssst, nrowssst, eType=gdal.GDT_Float32)
    dataset1.SetProjection(proj)
    dataset2.SetProjection(proj)
    dataset1.SetGeoTransform(geotransform)
    dataset2.SetGeoTransform(geotransform)
    _ = dataset1.GetRasterBand(1).WriteArray(np.flip(r_matrix, axis=0))
    _ = dataset2.GetRasterBand(1).WriteArray(np.flip(p_matrix, axis=0))
    dataset1.FlushCache()
    dataset2.FlushCache()
    dataset1 = None
    dataset2 = None
    # write correlation basins image
    corgrp_matrix[corgrp_matrix == -1] = 255
    corgrp_matrix[np.isnan(corgrp_matrix)] = 255
    output = outdir + os.sep + prefix + '_correlation-basins.tif'
    dataset3 = driver.Create(output, ncolssst, nrowssst, eType=gdal.GDT_Byte)
    dataset3.SetProjection(proj)
    dataset3.SetGeoTransform(geotransform)
    _ = dataset3.GetRasterBand(1).WriteArray(np.flip(corgrp_matrix, axis=0))
    dataset3.FlushCache()
    dataset3 = None
    # print correlation csv
    csv = outdir + os.sep + prefix + '_correlation-basin-avgs.csv'
    corr_df.reset_index()
    corr_df.to_csv(csv)


# def lregression(prefixParam, predictant, sst, lats, lons, PValue, selectMode, includeScore, stepwisePvalue, outdir):
#     trainStartYear = int(prefixParam["startyr"])
#     trainEndYear = int(prefixParam["endyr"])
#     fcstYear = int(prefixParam["fcstYear"])
#     fcstPeriod = prefixParam["fcstPeriod"]
#     name = re.sub('[^a-zA-Z0-9]', '', prefixParam["station"])
#     prefix = prefixParam["Predictor"] + '_' + prefixParam["Param"] + '_' + prefixParam["PredictorMonth"] + '_' + \
#              str(prefixParam["startyr"]) + '-' + str(prefixParam["endyr"]) + '_' + name
#     years = [yr for yr in range(trainStartYear, trainEndYear + 1)]
#     nyears = len(years)
#     SSTclusterSize = 1000.
#     trainPredictant = predictant[:nyears]
#     trainSST = sst[:nyears]
#     pnotnull = np.isfinite(trainPredictant)
#     nyearssst, nrowssst, ncolssst = sst.shape
#     yearssst = [yr for yr in range(trainStartYear, (trainStartYear + nyearssst))]
#     nsst = sst[yearssst.index(fcstYear)]
#     lons2d, lats2d = np.meshgrid(lons, lats)
#     # calculate correlation
#     r_matrix = np.zeros((nrowssst, ncolssst))
#     p_matrix = np.zeros((nrowssst, ncolssst))
#     # calculate correlation
#     for row in range(nrowssst):
#         for col in range(ncolssst):
#             sstvals = np.array(trainSST[:, row][:, col], dtype=float)
#             try:
#                 notnull = pnotnull & np.isfinite(sstvals)
#                 r_matrix[row][col], p_matrix[row][col] = pearsonr(trainPredictant[notnull], sstvals[notnull])
#             except:
#                 pass
#     #corr = (p_matrix <= PValue) & (abs(r_matrix) >= 0.5)
#     corr = (p_matrix <= PValue)
#     if not corr.any():
#         return 0
#     corr_coords = list(zip(lons2d[corr], lats2d[corr]))
#     # create correlation basins
#     corgrp_matrix = np.zeros((nrowssst, ncolssst)) * np.nan
#
#     minx = 0
#     maxx = 366
#     miny = -90
#     maxy = 90
#     roi = [False] * len(corr_coords)
#     for i in range(len(corr_coords)):
#         if corr_coords[i][0] < minx or corr_coords[i][0] > maxx or corr_coords[i][1] < miny or corr_coords[i][1] > maxy:
#             roi[i] = True
#
#     db = dbcluster(corr_coords, 'dbscan', 5, SSTclusterSize, 3, 2)
#     coords_clustered = np.array(db.labels_)
#     coords_clustered[roi] = -1
#     uniq = list(set(coords_clustered))
#     minpixelperbasin = 6
#     # if len(uniq) <= 15:
#     #     minpixelperbasin = 6
#     # else:
#     #     minpixelperbasin = 13
#     for zone in uniq:
#         count = len(coords_clustered[coords_clustered == zone])
#         if count < minpixelperbasin: coords_clustered[coords_clustered == zone] = -1
#
#     basins = list(set(coords_clustered[coords_clustered != -1]))
#     if len(basins) == 0: return None, None, None, None, None, None, None, None
#     SSTzones = len(basins)
#     if corr[corr == True].shape == coords_clustered.shape:
#         index = 0
#         for row in range(nrowssst):
#             for col in range(ncolssst):
#                 if corr[row][col]:
#                     corgrp_matrix[row][col] = coords_clustered[index]
#                     index = index + 1
#     # generate correlation group matrices
#     basin_arr = ['Basin' + str(x) for x in basins]
#     basin_arr.insert(0, fcstPeriod)
#     basin_arr.insert(0, 'year')
#     corr_df = pd.DataFrame(columns=basin_arr)
#     corr_df['year'] = years
#     corr_df[fcstPeriod] = trainPredictant
#     corr_df.set_index('year', inplace=True)
#     basin_matrix = np.zeros((nyearssst, SSTzones))
#     for yr in range(nyearssst):
#         year = yearssst[yr]
#         sstavg = np.zeros(SSTzones)
#         for group in range(SSTzones):
#             sstavg[group] = "{0:.3f}".format(np.mean(sst[yr][corgrp_matrix == basins[group]]))
#             corr_df.loc[year, 'Basin' + str(basins[group])] = sstavg[group]
#             basin_matrix[yr][group] = sstavg[group]
#     corroutdir = outdir + os.sep + "correlation"
#     # writeout(prefix, p_matrix, corgrp_matrix, corr_df, lats, lons, corroutdir)
#     writeout(prefix, r_matrix, p_matrix, corgrp_matrix, corr_df, lats, lons, corroutdir)
#     regoutdir = outdir + os.sep + "regression"
#     os.makedirs(regoutdir, exist_ok=True)
#     # get basin combination with highest r-square: returns bestr2score, final_basins, final_basin_matrix
#     print('checking best basin combination...')
#     if selectMode == 1:
#         try:
#             r2score, basin_arr, final_forecasts, basin_matrix, r2scores_df, regrFormula, comment_df = \
#                 best_basins_model(basins, basin_matrix, predictant, years, yearssst, includeScore, stepwisePvalue)
#             csv = regoutdir + os.sep + prefix + '_forward-selection.csv'
#             comment_df.to_csv(csv, header=True, index=False)
#             # LassoScore_df.to_csv(csv, mode='a', header=True, index=False)
#         except:
#             r2score, basin_arr, final_forecasts, basin_matrix, r2scores_df, regrFormula = best_basins(basins, basin_matrix,
#                                                                                            predictant, years, yearssst,
#                                                                                            includeScore)
#     else:
#         r2score, basin_arr, final_forecasts, basin_matrix, r2scores_df, regrFormula = best_basins(basins, basin_matrix,
#                                                                                                   predictant, years,
#                                                                                                   yearssst, includeScore)
#     # write basin combination scores to csv
#     csv = regoutdir + os.sep + prefix + '_basin-combinations.csv'
#     r2scores_df.sort_values('r2_score', axis=0, ascending=True, inplace=True, na_position='last')
#     r2scores_df.reset_index()
#     r2scores_df.to_csv(csv, index=False)
#     # write regression formula to file
#     selected_basins = ['Basin' + str(x) for x in basin_arr]
#     selected_basins.insert(0, 'y_intercept')
#     coeff_arr = list(regrFormula["coefficients"])
#     coeff_arr.insert(0, regrFormula["intercept"])
#     reg_df = pd.DataFrame(columns=selected_basins)
#     reg_df.loc[0] = coeff_arr
#     csv = regoutdir + os.sep + prefix + '_correlation-formula.csv'
#     reg_df.to_csv(csv, index=False)
#     # write forecasts to file
#     fcstColumns = ['Year', fcstPeriod, 'Forecast']
#     fcst_df = pd.DataFrame(columns=fcstColumns)
#     fcst_df['Year'] = yearssst
#     fcst_df[fcstPeriod] = predictant[:nyearssst]
#     fcst_df['Forecast'] = final_forecasts[:nyearssst]
#     fcst_df.set_index('Year', inplace=True)
#     csv = regoutdir + os.sep + prefix + '_yearly-forecasts.csv'
#     fcst_df.reset_index()
#     fcst_df.to_csv(csv, index=True)
#     # generate model skill statistics and write to file
#     limits = [0.3333,0.6667]
#     tercileyears = nyearssst
#     if not includeScore:
#         nrecentyears = nyearssst - nyears
#         fcst_df = fcst_df[-nrecentyears:]
#         tercileyears = nyears
#     t1, t2 = list(fcst_df[fcstPeriod][:tercileyears].quantile(limits))
#     fcst_df = fcst_df.rename(columns={fcstPeriod: "obs", "Forecast": "fcst"})
#     HS, HSS, POD_below, POD_normal, POD_above, FA_below, FA_normal, FA_above, cgtable_df, PB, PN, PA = \
#         model_skill(fcst_df, t1, t2)
#     skillColumns = ['Statistic', 'Value']
#     skill_df = pd.DataFrame(columns=skillColumns)
#     skill_df['Statistic'] = ['R-Squared Score', 'Hit Score (HS)', 'Hit Skill Score (HSS)',
#                              'Probability of Detecting Below', 'Probability of Detecting Normal',
#                              'Probability of Detecting Above', 'False Alarm 1st Order (Below)',
#                              'False Alarm 1st Order (Normal)', 'False Alarm 1st Order (Above)',
#                              'Probability Forecast For Below-Normal', 'Probability Forecast For Near-Normal',
#                              'Probability Forecast For Above-Normal']
#     skill_df['Value'] = [r2score, HS, HSS, POD_below, POD_normal, POD_above, FA_below, FA_normal, FA_above, PB, PN, PA]
#     csv = regoutdir + os.sep + prefix + '_score-contingency-table.csv'
#     cgtable_df.to_csv(csv, index=False)
#     csv = regoutdir + os.sep + prefix + '_score-statistics.csv'
#     skill_df.to_csv(csv, index=False)
#     # classify forecast
#     fcst_precip = round(final_forecasts[yearssst.index(fcstYear)], 1)
#     if fcst_precip < 0:  fcst_precip = 0.0
#     qlimits = [0.33, 0.5, 0.66]
#     dfp = pd.DataFrame(trainPredictant[pnotnull], columns=['avgrain']).quantile(qlimits)
#     q1 = float(round(dfp.loc[qlimits[0]], 1))
#     q2 = float(round(dfp.loc[qlimits[1]], 1))
#     q3 = float(round(dfp.loc[qlimits[2]], 1))
#     forecast_class = np.nan
#     Prob = np.nan
#     if fcst_precip <= q1:
#         forecast_class = 1
#         Prob = PB
#     if fcst_precip >= q1:
#         forecast_class = 2
#         Prob = PN
#     if fcst_precip >= q2:
#         forecast_class = 3
#         Prob = PN
#     if fcst_precip >= q3:
#         forecast_class = 4
#         Prob = PA
#     pmean = round(np.mean(trainPredictant[pnotnull]), 1)
#     return q1, q2, q3, pmean, fcst_precip, forecast_class, HS, str(Prob).replace(',',':')


# def forecast_point(predictor, param, predictorMonth, fcstPeriodType, station, station_data_all, sst_arr, lats, lons,
#                    trainStartYear, trainEndYear, predictorStartYr, fcstYear, fcstPeriod, PValue, composition,
#                    selectMode, includeScore, stepwisePvalue, outDir):
#     predictorName = os.path.splitext(os.path.basename(predictor))[0]
#     prefixParam = {"Predictor": predictorName, "Param": param, "PredictorMonth": predictorMonth, "startyr": trainStartYear,
#                    "endyr": trainEndYear, "fcstYear": fcstYear, "fcstPeriod": fcstPeriod, "station": str(station)}
#     years = [yr for yr in range(trainStartYear, trainEndYear + 1)]
#     nyearssst, nrowssst, ncolssst = sst_arr.shape
#     yearssst = [yr for yr in range(predictorStartYr, (predictorStartYr + nyearssst))]
#     yearspredictant = [yr for yr in range(trainStartYear, (trainStartYear + nyearssst))]
#     station_data = station_data_all.loc[:,
#                    ('Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')]
#     station_data.drop_duplicates('Year', inplace=True)
#     station_data = station_data.apply(pd.to_numeric, errors='coerce')
#     seasonal_precip = pd.DataFrame(columns=['Year',fcstPeriod])
#     seasonal_precip['Year'] = yearspredictant
#     seasonal_precip.set_index('Year', inplace=True)
#     station_data.set_index('Year', inplace=True)
#     for year in yearspredictant:
#         if fcstPeriodType == 0:
#             if composition == "Cumulation":
#                 seasonal_precip.loc[[year], fcstPeriod] = season_cumulation(station_data, year, fcstPeriod)
#             else:
#                 seasonal_precip.loc[[year], fcstPeriod] = season_average(station_data, year, fcstPeriod)
#         else:
#             try:
#                 seasonal_precip.loc[[year], fcstPeriod] = round(float(station_data.loc[[year], fcstPeriod]), 1)
#             except KeyError:
#                 seasonal_precip.loc[[year], fcstPeriod] = np.nan
#
#     station_precip = np.asarray(seasonal_precip, dtype=float).reshape(-1, )
#     # print(prefixParam, station_precip, sst_arr, lats, lons, PValue, selectMode, includeScore,
#     #                    stepwisePvalue, outDir)
#     return lregression(prefixParam, station_precip, sst_arr, lats, lons, PValue, selectMode, includeScore,
#                        stepwisePvalue, outDir)
#
#
# def forecast_zone(predictor, param, predictorMonth, fcstPeriodType, zone, zonestation, station_data_all, sst_arr,
#                   lats, lons, trainStartYear, trainEndYear, predictorStartYr, fcstYear,
#                   fcstPeriod, PValue, composition, selectMode, includeScore, stepwisePvalue, outDir):
#     precip_dfs = []
#     predictorName = os.path.splitext(os.path.basename(predictor))[0]
#     prefixParam = {"Predictor": predictorName, "Param": param, "PredictorMonth": predictorMonth, "startyr": trainStartYear,
#                    "endyr": trainEndYear, "fcstYear": fcstYear, "fcstPeriod": fcstPeriod, "station": str(zone)}
#     years = [yr for yr in range(trainStartYear, trainEndYear + 1)]
#     nyearssst, nrowssst, ncolssst = sst_arr.shape
#     yearssst = [yr for yr in range(predictorStartYr, (predictorStartYr + nyearssst))]
#     yearspredictant = [yr for yr in range(trainStartYear, (trainStartYear + nyearssst))]
#     for station in zonestation[zone]:
#         nzonestations = len(zonestation[zone])
#         station_data = station_data_all.loc[station_data_all['ID'] == station]
#         station_data = station_data.loc[:,
#                        ('Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')]
#         station_data.drop_duplicates('Year', inplace=True)
#         station_data = station_data.apply(pd.to_numeric, errors='coerce')
#         station_data.set_index('Year', inplace=True)
#         seasonal_precip = pd.DataFrame(columns=['Year', fcstPeriod])
#         seasonal_precip['Year'] = yearspredictant
#         seasonal_precip.set_index('Year', inplace=True)
#         for year in yearspredictant:
#             if fcstPeriodType == 0:
#                 if composition == "Cumulation":
#                     seasonal_precip.loc[[year], fcstPeriod] = season_cumulation(station_data, year, fcstPeriod)
#                 else:
#                     seasonal_precip.loc[[year], fcstPeriod] = season_average(station_data, year, fcstPeriod)
#             else:
#                 try:
#                     seasonal_precip.loc[[year], fcstPeriod] = round(float(station_data.loc[[year], fcstPeriod]), 1)
#                 except KeyError:
#                     seasonal_precip.loc[[year], fcstPeriod] = np.nan
#         precip_dfs.append(seasonal_precip)
#     precip_concat = pd.concat((precip_dfs), axis=1)
#     zone_precip = np.asarray(precip_concat.mean(axis=1)).reshape(-1, )
#     return lregression(prefixParam, zone_precip, sst_arr, lats, lons, PValue, selectMode, includeScore,
#                        stepwisePvalue, outDir)
#
#
# def forecast_point(station_id):
#     global config
#     global predictordict
#     global fcstPeriod
#     global predictantdict
#     station = predictantdict['stations'][station_id]
#     input_data = predictantdict['data']
#     station_data_all =  input_data.loc[input_data['ID'] == station]
#     outDir = config.get('outDir') + os.sep + 'Forecast_' + str(config.get('fcstyear')) + \
#               '_' + fcstPeriod + os.sep + 'LR'
#     for predictorName in predictordict:
#         predictorStartYr = predictordict[predictorName]['predictorStartYr']
#         sst_arr = predictordict[predictorName]['sst_arr']
#         trainStartYear = config['trainStartYear']
#         fcstPeriodType = config['fcstPeriodType']
#         prefixParam = {"Predictor": predictorName, "Param": predictordict[predictorName]['param'],
#                        "PredictorMonth": predictordict[predictorName]['predictorMonth'],
#                        "startyr": trainStartYear, "endyr": config['trainEndYear'],
#                        "fcstYear": config['fcstyear'], "fcstPeriod": fcstPeriod, "station": str(station)}
#         years = [yr for yr in range(int(config['trainStartYear']), int(config['trainEndYear']) + 1)]
#         nyearssst, nrowssst, ncolssst = sst_arr.shape
#         yearssst = [yr for yr in range(predictorStartYr, (predictorStartYr + nyearssst))]
#         yearspredictant = [yr for yr in range(trainStartYear, (trainStartYear + nyearssst))]
#         station_data = station_data_all.loc[:,
#                        ('Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')]
#         station_data.drop_duplicates('Year', inplace=True)
#         station_data = station_data.apply(pd.to_numeric, errors='coerce')
#         seasonal_precip = pd.DataFrame(columns=['Year',fcstPeriod])
#         seasonal_precip['Year'] = yearspredictant
#         seasonal_precip.set_index('Year', inplace=True)
#         station_data.set_index('Year', inplace=True)
#         for year in yearspredictant:
#             if fcstPeriodType == 0:
#                 if composition == "Cumulation":
#                     seasonal_precip.loc[[year], fcstPeriod] = season_cumulation(station_data, year, fcstPeriod)
#                 else:
#                     seasonal_precip.loc[[year], fcstPeriod] = season_average(station_data, year, fcstPeriod)
#             else:
#                 try:
#                     seasonal_precip.loc[[year], fcstPeriod] = round(float(station_data.loc[[year], fcstPeriod]), 1)
#                 except KeyError:
#                     seasonal_precip.loc[[year], fcstPeriod] = np.nan
#
#         station_precip = np.asarray(seasonal_precip, dtype=float).reshape(-1, )
#         # print(prefixParam, station_precip, sst_arr, lats, lons, PValue, selectMode, includeScore,
#         #                    stepwisePvalue, outDir)
#     return lregression(prefixParam, station_precip, sst_arr, lats, lons, PValue, selectMode, includeScore,
#                        stepwisePvalue, outDir)
#
def run_model_skill(fcst_df, fcstPeriod, fcstcol, r2score, training_actual):
    # generate model skill statistics and write to file
    limits = [0.3333, 0.6667]
    fcst_precip = round(list(fcst_df.tail(1)[fcstcol])[0], 1)
    observed = np.array(training_actual)
    observed_clean = observed[np.isfinite(observed)]
    t1, t2 = list(pd.DataFrame(training_actual)[0].quantile(limits))
    nfcst_df = fcst_df.rename(columns={fcstPeriod: "obs", fcstcol: "fcst"})
    HS, HSS, POD_below, POD_normal, POD_above, FA_below, FA_normal, FA_above, cgtable_df, PB, PN, PA = \
        model_skill(nfcst_df, t1, t2)
    skillColumns = ['Statistic', 'Value']
    skill_df = pd.DataFrame(columns=skillColumns)
    skill_df['Statistic'] = ['R-Squared Score', 'Hit Score (HS)', 'Hit Skill Score (HSS)',
                             'Probability of Detecting Below', 'Probability of Detecting Normal',
                             'Probability of Detecting Above', 'False Alarm 1st Order (Below)',
                             'False Alarm 1st Order (Normal)', 'False Alarm 1st Order (Above)',
                             'Probability Forecast For Below-Normal', 'Probability Forecast For Near-Normal',
                             'Probability Forecast For Above-Normal']
    skill_df['Value'] = [r2score, HS, HSS, POD_below, POD_normal, POD_above, FA_below, FA_normal, FA_above, PB,
                         PN, PA]
    # csv = regoutdir + os.sep + prefix + '_score-contingency-table.csv'
    # cgtable_df.to_csv(csv, index=False)
    # csv = regoutdir + os.sep + prefix + '_score-statistics.csv'
    # skill_df.to_csv(csv, index=False)
    # classify forecast
    if fcst_precip < 0:  fcst_precip = 0.0
    qlimits = [0.33, 0.5, 0.66]
    dfp = pd.DataFrame(observed_clean)[0].quantile(qlimits)
    q1 = float(round(dfp.loc[qlimits[0]], 1))
    q2 = float(round(dfp.loc[qlimits[1]], 1))
    q3 = float(round(dfp.loc[qlimits[2]], 1))
    forecast_class = np.nan
    Prob = np.nan
    if fcst_precip <= q1:
        forecast_class = 1
        Prob = PB
    if fcst_precip >= q1:
        forecast_class = 2
        Prob = PN
    if fcst_precip >= q2:
        forecast_class = 3
        Prob = PN
    if fcst_precip >= q3:
        forecast_class = 4
        Prob = PA
    pmean = round(np.mean(observed_clean), 1)
    return q1, q2, q3, pmean, fcst_precip, forecast_class, HS, str(Prob).replace(',', ':'), cgtable_df, skill_df


def forecast_station(config, predictordict, predictantdict, fcstPeriod, outdir, station):
    output = {}
    stationYF_dfs = []
    output[station] = {}
    input_data = predictantdict['data']
    indx = predictantdict['stations'].index(station)
    lat = predictantdict['lats'][indx]
    lon = predictantdict['lons'][indx]
    station_data_all =  input_data.loc[input_data['ID'] == station]
    trainStartYear = int(config['trainStartYear'])
    trainEndYear = int(config['trainEndYear'])
    fcstPeriodType = int(config['fcstPeriodType'])
    fcstYear = int(config['fcstyear'])
    forecastdf = pd.DataFrame(columns=['Predictor', 'Algorithm', 'ID', 'Lat', 'Lon', 't1', 't2', 't3',
                                       'mean', 'fcst', 'class', 'r2score', 'HS', 'Prob'])

    for predictorName in predictordict:
        print('\npredictor',predictorName,'...')
        output[station][predictorName] = {}
        predictorStartYr = predictordict[predictorName]['predictorStartYr']
        sst_arr = predictordict[predictorName]['data']
        prefixParam = {"Predictor": predictorName, "Param": predictordict[predictorName]['param'],
                       "PredictorMonth": predictordict[predictorName]['predictorMonth'],
                       "startyr": trainStartYear, "endyr": config['trainEndYear'],
                       "fcstYear": config['fcstyear'], "fcstPeriod": fcstPeriod, "station": str(station)}
        # years = [yr for yr in range(int(config['trainStartYear']), int(config['trainEndYear']) + 1)]
        nyearssst, nrowssst, ncolssst = sst_arr.shape
        # yearssst = [yr for yr in range(predictorStartYr, (predictorStartYr + nyearssst))]
        yearspredictant = [yr for yr in range(trainStartYear, (trainStartYear + nyearssst))]
        station_data = station_data_all.loc[:,
                       ('Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')]
        station_data.drop_duplicates('Year', inplace=True)
        station_data = station_data.apply(pd.to_numeric, errors='coerce')
        seasonal_precip = pd.DataFrame(columns=['Year',fcstPeriod])
        seasonal_precip['Year'] = yearspredictant
        seasonal_precip.set_index('Year', inplace=True)
        station_data.set_index('Year', inplace=True)
        for year in yearspredictant:
            if fcstPeriodType == 0:
                if config['composition'] == "Cumulation":
                    seasonal_precip.loc[[year], fcstPeriod] = season_cumulation(station_data, year, fcstPeriod)
                else:
                    seasonal_precip.loc[[year], fcstPeriod] = season_average(station_data, year, fcstPeriod)
            else:
                try:
                    seasonal_precip.loc[[year], fcstPeriod] = round(float(station_data.loc[[year], fcstPeriod]), 1)
                except KeyError:
                    seasonal_precip.loc[[year], fcstPeriod] = np.nan

        predictant = np.asarray(seasonal_precip, dtype=float).reshape(-1, )

        ########## compute basins #############
        name = re.sub('[^a-zA-Z0-9]', '', prefixParam["station"])
        prefix = prefixParam["Predictor"] + '_' + prefixParam["Param"] + '_' + prefixParam["PredictorMonth"] + '_' + \
                 str(prefixParam["startyr"]) + '-' + str(prefixParam["endyr"]) + '_' + name
        trainingYears = [yr for yr in range(trainStartYear, trainEndYear + 1)]
        nyears = len(trainingYears)
        SSTclusterSize = 1000.
        trainPredictant = predictant[:nyears]
        trainSST = sst_arr[:nyears]
        pnotnull = np.isfinite(trainPredictant)
        nyearssst, nrowssst, ncolssst = sst_arr.shape
        yearssst = [yr for yr in range(trainStartYear, (trainStartYear + nyearssst))]
        # nsst = sst_arr[yearssst.index(fcstYear)]
        lons2d, lats2d = np.meshgrid(predictordict[predictorName]['lons'], predictordict[predictorName]['lats'])
        # calculate correlation
        r_matrix = np.zeros((nrowssst, ncolssst))
        p_matrix = np.zeros((nrowssst, ncolssst))
        # calculate correlation
        for row in range(nrowssst):
            for col in range(ncolssst):
                sstvals = np.array(trainSST[:, row][:, col], dtype=float)
                warnings.filterwarnings('error')
                try:
                    notnull = pnotnull & np.isfinite(sstvals)
                    r_matrix[row][col], p_matrix[row][col] = pearsonr(trainPredictant[notnull], sstvals[notnull])
                except:
                    pass
        # corr = (p_matrix <= PValue) & (abs(r_matrix) >= 0.5)
        corr = (p_matrix <= config['PValue'])
        if not corr.any():
            return 0
        corr_coords = list(zip(lons2d[corr], lats2d[corr]))
        # create correlation basins
        corgrp_matrix = np.zeros((nrowssst, ncolssst)) * np.nan

        minx = 0
        maxx = 366
        miny = -90
        maxy = 90
        roi = [False] * len(corr_coords)
        for i in range(len(corr_coords)):
            if corr_coords[i][0] < minx or corr_coords[i][0] > maxx or corr_coords[i][1] < miny or \
                    corr_coords[i][1] > maxy:
                roi[i] = True

        db = dbcluster(corr_coords, 'dbscan', 5, SSTclusterSize, 3, 2)
        coords_clustered = np.array(db.labels_)
        coords_clustered[roi] = -1
        uniq = list(set(coords_clustered))
        minpixelperbasin = 6
        # if len(uniq) <= 15:
        #     minpixelperbasin = 6
        # else:
        #     minpixelperbasin = 13
        for zone in uniq:
            count = len(coords_clustered[coords_clustered == zone])
            if count < minpixelperbasin: coords_clustered[coords_clustered == zone] = -1

        basins = list(set(coords_clustered[coords_clustered != -1]))
        if len(basins) == 0: return None, None, None, None, None, None, None, None
        SSTzones = len(basins)
        if corr[corr == True].shape == coords_clustered.shape:
            index = 0
            for row in range(nrowssst):
                for col in range(ncolssst):
                    if corr[row][col]:
                        corgrp_matrix[row][col] = coords_clustered[index]
                        index = index + 1
        # generate correlation group matrices
        basin_arr = ['Basin' + str(x) for x in basins]
        basin_arr.insert(0, fcstPeriod)
        basin_arr.insert(0, 'year')
        corr_df = pd.DataFrame(columns=basin_arr)
        corr_df['year'] = trainingYears
        corr_df[fcstPeriod] = trainPredictant
        corr_df.set_index('year', inplace=True)
        for yr in range(nyearssst):
            year = yearssst[yr]
            sstavg = np.zeros(SSTzones)
            corr_df.loc[year, fcstPeriod] = list(seasonal_precip.loc[[year], fcstPeriod])[0]
            for group in range(SSTzones):
                sstavg[group] = "{0:.3f}".format(np.mean(sst_arr[yr][corgrp_matrix == basins[group]]))
                corr_df.loc[year, 'Basin' + str(basins[group])] = sstavg[group]
        corr_df = corr_df.dropna(how='all', axis=1)
        basin_arr = list(corr_df.columns)
        indx = basin_arr.index(fcstPeriod)
        basin_arr.pop(indx)
        basins = [x.replace('Basin','') for x in basin_arr]
        basin_matrix = np.array(corr_df[basin_arr])
        corroutdir = outdir + os.sep + "Correlation"
        writeout(prefix, r_matrix, p_matrix, corgrp_matrix, corr_df, predictordict[predictorName]['lats'],
                 predictordict[predictorName]['lons'], corroutdir)

        # get basin combination with highest r-square: returns bestr2score, final_basins, final_basin_matrix
        print('checking best basin combination...')
        basin_matrix_df = pd.DataFrame(basin_matrix[:len(trainingYears)], columns=basin_arr)
        notnull = np.isfinite(np.array(predictant[:len(trainingYears)]))
        final_basins, comments = stepwise_selection(basin_matrix_df[notnull].astype(float),
                                                    list(predictant[:len(trainingYears)][notnull]),
                                                    initial_list=basin_arr, threshold_out=config['stepwisePvalue'])

        selected_basins = final_basins[:]
        selected_basins.insert(0, 'y_intercept')
        comments.append("Final basins: " + str(final_basins))
        csv = corroutdir + os.sep + prefix + '_forward-selection.csv'
        comment_df = pd.DataFrame(columns=['Comment'])
        comment_df['Comment'] = comments
        comment_df.to_csv(csv, header=True, index=False)

        combo_basin_matrix = np.zeros((len(yearssst), len(final_basins))) * np.nan
        # loop for all years where SST is available
        for yr in range(len(yearssst)):
            for group in range(len(final_basins)):
                # get corresponding sst average for the group from main basin_matrix
                combo_basin_matrix[yr][group] = basin_matrix[yr][basin_arr.index(final_basins[group])]

        nbasins = len(final_basins)
        training_actual = predictant[:len(trainingYears)]
        test_actual = predictant[len(trainingYears):]
        training_Xmatrix = combo_basin_matrix[:len(trainingYears)]
        testing_Xmatrix = combo_basin_matrix[len(trainingYears):]
        testing_years = yearssst[len(trainingYears):]
        notnull = np.isfinite(training_actual)
        test_notnull = np.isfinite(test_actual)
        # scale the predictor
        scaler = StandardScaler()
        scaler.fit(training_Xmatrix)
        StandardScaler(copy=True, with_mean=True, with_std=True)
        X_train = scaler.transform(training_Xmatrix)
        X_test = scaler.transform(testing_Xmatrix)

        regoutdir = outdir + os.sep + "Regression"
        os.makedirs(regoutdir, exist_ok=True)

        for algorithm in config.get('algorithms'):
            if algorithm == 'LR':
                print('linear regression...')
                # start_time = time.time()
                regr = linear_model.LinearRegression()
                regr.fit(X_train[notnull], np.asarray(training_actual)[notnull])
                intercept = regr.intercept_
                coefficients = regr.coef_
                lr_fcstdf = pd.DataFrame(columns=['Year', fcstPeriod, 'LRfcst'])
                lr_fcstdf['Year'] = testing_years
                lr_fcstdf[fcstPeriod] = test_actual
                lr_fcstdf.set_index('Year', inplace=True)
                for yr in range(len(testing_years)):
                    year = testing_years[yr]
                    lr_fcstdf.loc[year, 'LRfcst'] = regr.predict(X_test[yr].reshape(1, -1))[0]
                warnings.filterwarnings('error')
                m, n = pearsonr(np.array(lr_fcstdf['LRfcst'])[test_notnull], list(np.ravel(test_actual)[test_notnull]))
                r2score = m ** 2
                # print('r-square', r2score, ', p', n)
                # print('processing time(sec)', time.time() - start_time)
                lrdirout = regoutdir + os.sep + 'LR'
                os.makedirs(lrdirout, exist_ok=True)
                csv = lrdirout + os.sep + prefix + '_' + fcstPeriod + '_forecast_matrix.csv'
                lr_fcstdf.reset_index()
                lr_fcstdf.to_csv(csv, index=True)
                #
                regrFormula = {"intercept": intercept, "coefficients": coefficients}
                coeff_arr = list(regrFormula["coefficients"])
                coeff_arr.insert(0, regrFormula["intercept"])
                reg_df = pd.DataFrame(columns=selected_basins)
                reg_df.loc[0] = coeff_arr
                csv = lrdirout+ os.sep + prefix + '_correlation-formula.csv'
                reg_df.to_csv(csv, index=False)
                #
                q1, q2, q3, pmean, famnt, fclass, HS, Prob, cgtable_df, skill_df = \
                    run_model_skill(lr_fcstdf, fcstPeriod, 'LRfcst', r2score, training_actual)
                csv = lrdirout + os.sep + prefix + '_score-contingency-table.csv'
                cgtable_df.to_csv(csv, index=False)
                csv = lrdirout + os.sep + prefix + '_score-statistics.csv'
                skill_df.to_csv(csv, index=False)
                a_series = pd.Series([predictorName, algorithm, station, lat, lon, q1, q2, q3, pmean, famnt,
                                      fclass, r2score, HS, Prob], index=forecastdf.columns)
                forecastdf = forecastdf.append(a_series, ignore_index=True)
                lr_fcstdf.rename(columns={'LRfcst': predictorName + '_LR'}, inplace=True)
                stationYF_dfs.append(lr_fcstdf)

            if algorithm == 'MLP':
                print('mlp regression...')
                start_time = time.time()
                activation_fn = 'tanh'
                solver_fn = 'lbfgs'
                ratings = {}
                for x in range(50):
                    for y in range(1, 20):
                        if y > x: continue
                        forecasts = []
                        if y == 0:
                            hiddenlayerSize = (x + 1,)
                        else:
                            if x > 20: continue
                            hiddenlayerSize = (x + 1, y + 1)
                        regm = MLPRegressor(hidden_layer_sizes=hiddenlayerSize,
                                            activation=activation_fn, solver=solver_fn, random_state=42, max_iter=700)
                        regm.fit(X_train[notnull], np.asarray(training_actual)[notnull])
                        for z in range(len(X_test)):
                            forecasts.append(regm.predict(X_test[z].reshape(1, -1))[0])
                        warnings.filterwarnings('error')
                        try:
                            m, n = pearsonr(np.array(forecasts)[test_notnull], list(np.ravel(test_actual)[test_notnull]))
                        except:
                            pass
                        v = np.std(forecasts)
                        ratings[str(x + 1) + '_' + str(y + 1)] = (m ** 2, v)

                combs = sorted(ratings.items(), key=lambda xx: xx[1][0], reverse=True)
                v = np.std(np.ravel(test_actual[test_notnull]))
                for x in range(len(combs)):
                    if combs[x][1][0] >= 0.1 and combs[x][1][1] >= v / 2:
                        r, s = combs[x][0].split('_')
                        break
                if int(s) == 0:
                    hiddenlayerSize = (int(r),)
                else:
                    hiddenlayerSize = (int(r), int(s),)
                regm = MLPRegressor(hidden_layer_sizes=hiddenlayerSize,
                                    activation=activation_fn, solver=solver_fn, random_state=42, max_iter=700)
                regm.fit(X_train[notnull], np.asarray(training_actual)[notnull])
                mlp_fcstdf = pd.DataFrame(columns=['Year', fcstPeriod, 'MLPfcst'])
                mlp_fcstdf['Year'] = testing_years
                mlp_fcstdf[fcstPeriod] = test_actual
                mlp_fcstdf.set_index('Year', inplace=True)
                for yr in range(len(testing_years)):
                    year = testing_years[yr]
                    mlp_fcstdf.loc[year, 'MLPfcst'] = regm.predict(X_test[yr].reshape(1, -1))[0]
                warnings.filterwarnings('error')
                m, n = pearsonr(np.array(mlp_fcstdf['MLPfcst'])[test_notnull], list(np.ravel(test_actual[test_notnull])))
                r2score = m ** 2
                mlpdirout = regoutdir + os.sep + 'MLP'
                os.makedirs(mlpdirout, exist_ok=True)
                file = mlpdirout + os.sep + prefix + '_' + fcstPeriod + '_mlpsummary.txt'
                f = open(file, 'w')
                f.write('MLPRegressor Parameters ---\n')
                f.write('architecture=' + str(nbasins) + ',' + r + ',' + s + ',1\n')
                f.write('r-square: ' + str(r2score) + ', p-value:' + str(n) + '\n')
                f.write('processing time: ' + str(time.time() - start_time) + ' seconds\n\n')
                f.write(json.dumps(regm.get_params(), indent=4, sort_keys=True))
                f.write('\n\n')
                f.write('Ranking of number of neurons per hidden layer (HL) ---\n')
                f.write('("HL1_HL2", (r2score, std))\n')
                for ele in combs[:20]:
                    f.write(str(ele) + '\n')
                f.close()
                csv = mlpdirout + os.sep + prefix + '_' + fcstPeriod + '_forecast_matrix.csv'
                mlp_fcstdf.reset_index()
                mlp_fcstdf.to_csv(csv, index=True)
                #
                q1, q2, q3, pmean, famnt, fclass, HS, Prob, cgtable_df, skill_df = \
                    run_model_skill(mlp_fcstdf, fcstPeriod, 'MLPfcst', r2score, training_actual)
                csv = mlpdirout + os.sep + prefix + '_score-contingency-table.csv'
                cgtable_df.to_csv(csv, index=False)
                csv = mlpdirout + os.sep + prefix + '_score-statistics.csv'
                skill_df.to_csv(csv, index=False)
                a_series = pd.Series([predictorName, algorithm, station, lat, lon, q1, q2, q3, pmean, famnt,
                                      fclass, r2score, HS, Prob], index=forecastdf.columns)
                forecastdf = forecastdf.append(a_series, ignore_index=True)
                mlp_fcstdf.rename(columns={'MLPfcst': predictorName +'_MLP'}, inplace=True)
                stationYF_dfs.append(mlp_fcstdf)

    # plot the forecast graphs
    stationYF_df = pd.concat(stationYF_dfs, axis=1, join='outer')
    stationYF_df = stationYF_df.loc[:, ~stationYF_df.columns.duplicated()]
    stationYF_df = stationYF_df.reset_index()
    fcstoutdir = outdir + os.sep + "Forecast"
    os.makedirs(fcstoutdir, exist_ok=True)
    graphpng = fcstoutdir + os.sep + 'forecast_graphs_' + station + '.png'
    plot_Station_forecast(stationYF_df, fcstPeriod, graphpng, station, q1, q2, q3)
    # return station forecast
    return forecastdf
